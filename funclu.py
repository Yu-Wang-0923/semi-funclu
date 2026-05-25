"""Functional Clustering 后端模块。

实现进度：

- ``_prepare_data``：多 condition DataFrame → ``(n_features, n_times_i)`` 张量列表。
- ``_initialize``：基于 KMeans / MiniBatchKMeans 给出 EM 的 4 件套初值
  ``(labels, weights, mu_params, cov_params)``，外加 ``centers_kl``（KMeans 子中心，
  调试/可视化用）与 ``backend``（实际使用的 KMeans 后端名）。
- ``_e_step / _m_step / fit``：基于 SAD1 协方差 + 幂律均值 ``μ = a · t^b`` 的 EM 主循环。
  M 步采用"半 EM"方案：``a`` 内层迭代闭式更新，``b`` 冻在上一轮；``gamma`` 用加权
  二次型闭式，``phi`` 冻在上一轮（NaN 兜底时回退到残差自相关重估）。
- ``predict`` / ``get_params`` / ``get_cluster_curves``：拟合完成后的辅助接口。

模型参数计数（用于 BIC）：``n_params = K · L · 4 + max(K - 1, 0)``，每个 ``(k, i)``
含 4 个参数 ``[a, b, phi, gamma]``，加 ``K - 1`` 个独立的混合权重。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

try:
    import torch
except ImportError:
    raise ImportError(
        "idopnetwork[ml] is required for clustering. "
        "Install with: pip install idopnetwork[ml]"
    )

from idopnetwork.curve_fitting import fit_power_loglinear


class FunClu:
    """多 condition 函数聚类（构建中）。

    Args:
        n_components: 簇数 K，默认 3。
        max_iter: EM 主循环最大迭代次数，默认 50（本步未启用）。
        tol: EM 收敛阈值（log-likelihood 增量），默认 1e-4（本步未启用）。
        device: 张量驻留设备，默认 ``torch.device('cpu')``。
        dtype: 张量精度，默认 ``torch.float64``。
        kmeans_minibatch_threshold: 当 ``n_features`` ≥ 该阈值且
            ``use_minibatch_kmeans is None`` 时自动启用 ``MiniBatchKMeans``，默认 8000。
        minibatch_batch_size: ``MiniBatchKMeans`` 的 batch 大小，默认 4096。
        minibatch_max_iter: ``MiniBatchKMeans`` 的最大迭代次数，默认 100。
        use_minibatch_kmeans: ``None`` 自动；``True`` 强制 MiniBatch；``False`` 强制全量。
        random_state: KMeans 随机种子，默认 42。
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 50,
        tol: float = 1e-4,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        *,
        kmeans_minibatch_threshold: int = 8_000,
        minibatch_batch_size: int = 4_096,
        minibatch_max_iter: int = 100,
        use_minibatch_kmeans: Optional[bool] = None,
        random_state: int = 42,
    ) -> None:
        self.n_components: int = int(n_components)
        self.max_iter: int = int(max_iter)
        self.tol: float = float(tol)
        self.device: torch.device = device if device is not None else torch.device("cpu")
        self.dtype: torch.dtype = dtype
        self.kmeans_minibatch_threshold: int = int(kmeans_minibatch_threshold)
        self.minibatch_batch_size: int = int(minibatch_batch_size)
        self.minibatch_max_iter: int = int(minibatch_max_iter)
        self.use_minibatch_kmeans: Optional[bool] = use_minibatch_kmeans
        self.random_state: int = int(random_state)

        # ── 数据维度（_prepare_data 写入） ────────────────────────────────────
        self.common_cols: Optional[List[str]] = None
        self.n_features: int = 0
        self.n_conditions: int = 0
        self.times_list: Optional[List[np.ndarray]] = None
        self.n_times_conditions: Optional[List[int]] = None

        # ── 模型参数（_initialize / EM 写入） ─────────────────────────────────
        self.params_mu: Optional[torch.Tensor] = None    # (K, L, 2) [a, b]
        self.params_cov: Optional[torch.Tensor] = None   # (K, L, 2) [phi, gamma]
        self.weights: Optional[torch.Tensor] = None      # (K,)

        # ── 结果（EM 写入） ───────────────────────────────────────────────────
        self.labels: Optional[torch.Tensor] = None
        self.log_likelihood: Optional[float] = None
        self.neg_log_likelihood: Optional[float] = None
        self.bic: Optional[float] = None
        self.n_params: Optional[int] = None
        self.converged: bool = False
        self.loglik_history: List[float] = []
        self.n_iter_run: int = 0
        self._kmeans_init_backend: str = ""

    # ────────────────────────────────────────────────────────────────────────
    # Step 1: data preparation
    # ────────────────────────────────────────────────────────────────────────
    def _prepare_data(
        self,
        data: List[pd.DataFrame],
    ) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
        """从多 condition 的 DataFrame 列表抽取张量与时间向量。

        各 condition 的时间长度可以不同，不做对齐；仅按列名取**交集**，
        确保每个 condition 在相同的特征列上参与聚类。

        Args:
            data: 长度为 ``n_conditions`` 的列表，第 i 个元素为
                ``(n_times_i, n_features_i)`` 的 ``pd.DataFrame``，
                行索引解释为时间/伪时间，列为特征。

        Returns:
            ``(X_list, times_list)``：

            - ``X_list``：长度为 ``n_conditions`` 的列表；第 i 项为形如
              ``(n_features, n_times_i)`` 的 ``torch.Tensor``，
              其中 ``n_features`` 等于所有 condition 的列名交集大小；
            - ``times_list``：长度为 ``n_conditions`` 的列表；第 i 项为
              ``(n_times_i,)`` 的 ``np.ndarray``（``float64``）。

        Raises:
            ValueError: 若 ``data`` 为空，或所有 condition 的列交集为空。
        """
        if not data:
            raise ValueError("data 不能为空：至少需要 1 个 condition 的 DataFrame")

        common_cols: List[str] = list(data[0].columns)
        for d in data[1:]:
            cols_d = set(d.columns)
            common_cols = [c for c in common_cols if c in cols_d]
        if len(common_cols) == 0:
            raise ValueError("所有 condition 的 DataFrame 不存在共同列；无法聚类")

        self.common_cols = common_cols
        self.n_features = len(common_cols)
        self.n_conditions = len(data)

        X_list: List[torch.Tensor] = []
        times_list: List[np.ndarray] = []
        for df in data:
            sub = df[common_cols]
            vals = sub.to_numpy(dtype=np.float64, copy=True).T
            idx = np.asarray(sub.index, dtype=np.float64)
            X_list.append(torch.from_numpy(vals).to(self.device, self.dtype))
            times_list.append(idx)

        self.times_list = times_list
        self.n_times_conditions = [len(t) for t in times_list]

        return X_list, times_list

    # ────────────────────────────────────────────────────────────────────────
    # Step 2: KMeans-based initialization
    # ────────────────────────────────────────────────────────────────────────
    def _initialize(
        self,
        X_list: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """基于 KMeans / MiniBatchKMeans 给出 EM 4 件套初值。

        流程：

        1. 沿时间轴拼接所有 condition：得到 ``(N, sum(n_t_i))`` 的输入矩阵。
        2. 选择 KMeans 后端（按 ``use_minibatch_kmeans`` 与
           ``kmeans_minibatch_threshold`` 自动切换）并跑一次。
        3. 把簇中心按 ``n_t_i`` 切回每个 ``(k, i)`` 的"子中心"。
        4. 对每个 ``(k, i)`` 用双对数线性回归拟合 ``y = a · t^b`` →
           ``params_mu[k, i] = (a, b)``，并把结果裁剪到
           ``a ∈ [1e-8, ∞)`` 与 ``b ∈ [-10, 10]``（允许衰减型 ``b < 0``，
           同时给极端值兜底）。
        5. 估计 SAD1 的 ``(phi, gamma)``：对该簇所有特征求残差均值序列
           ``r̄ = mean(X_ki - μ_ki)``；``phi`` 取一阶自相关并夹紧到
           ``[-0.99, 0.99]``；``gamma = sqrt(var(r̄) · (1 - phi²) + 1e-6)``。
           **空簇**或 ``n_t_i == 1`` 等退化情况按文档兜底（``phi=0, gamma=1``）。

        本方法**会写入** ``self.params_mu / params_cov / weights / labels /
        _kmeans_init_backend``，便于 EM 步直接使用；同时把上述结果与
        ``centers_kl / backend`` 一起作为字典返回。

        Args:
            X_list: ``_prepare_data`` 已经准备好的张量列表，长度 = ``n_conditions``，
                第 i 项形如 ``(n_features, n_times_i)``。

        Returns:
            字典，含以下键：

            - ``labels``：``torch.LongTensor``，形如 ``(n_features,)``。
            - ``weights``：``torch.Tensor``，形如 ``(K,)``，权重之和归一化为 1。
            - ``mu_params``：``torch.Tensor``，形如 ``(K, L, 2)``，最后一维为 ``[a, b]``。
            - ``cov_params``：``torch.Tensor``，形如 ``(K, L, 2)``，最后一维为
              ``[phi, gamma]``。
            - ``centers_kl``：``List[List[np.ndarray]]``，``centers_kl[k][i]`` 为
              该 ``(k, i)`` 的 KMeans 子中心，形如 ``(n_t_i,)``。
            - ``backend``：实际使用的 KMeans 后端名（``"KMeans"`` 或 ``"MiniBatchKMeans"``）。

        Raises:
            RuntimeError: 若尚未调用过 ``_prepare_data``（``n_conditions == 0``）。
        """
        if self.n_conditions == 0 or self.times_list is None or self.n_times_conditions is None:
            raise RuntimeError(
                "_initialize 调用前必须先调用 _prepare_data 设置 n_conditions / times_list"
            )
        if len(X_list) != self.n_conditions:
            raise ValueError(
                f"X_list 长度 {len(X_list)} 与 n_conditions {self.n_conditions} 不一致"
            )

        K = self.n_components
        L = self.n_conditions
        n_t_per: List[int] = list(self.n_times_conditions)

        # 1) 拼接：(N, sum(n_t_i))
        X_concat = torch.cat(X_list, dim=1)
        N = int(X_concat.shape[0])
        X_np = X_concat.detach().cpu().numpy()
        if K > N:
            raise ValueError(
                f"n_components={K} 超过特征数 N={N}；请减小 n_components"
            )

        # 2) KMeans 后端选择
        use_mb = self.use_minibatch_kmeans
        if use_mb is None:
            use_mb = N >= self.kmeans_minibatch_threshold

        if use_mb:
            bs = max(256, min(self.minibatch_batch_size, N))
            km = MiniBatchKMeans(
                n_clusters=K,
                init="k-means++",
                batch_size=bs,
                n_init=3,
                max_iter=self.minibatch_max_iter,
                random_state=self.random_state,
            ).fit(X_np)
            backend = "MiniBatchKMeans"
        else:
            km = KMeans(
                n_clusters=K,
                init="k-means++",
                n_init=10,
                random_state=self.random_state,
            ).fit(X_np)
            backend = "KMeans"

        labels_np: np.ndarray = km.labels_.astype(np.int64, copy=False)
        centers_concat: np.ndarray = km.cluster_centers_.astype(np.float64, copy=False)

        # 簇大小 / 权重
        sizes = np.bincount(labels_np, minlength=K)
        weights_np = sizes.astype(np.float64) / max(N, 1)
        if weights_np.sum() > 0:
            weights_np = weights_np / weights_np.sum()

        # 3) 切回 (k, i) 子中心
        centers_kl: List[List[np.ndarray]] = [
            [np.empty((0,), dtype=np.float64) for _ in range(L)] for _ in range(K)
        ]
        offset = 0
        for i in range(L):
            n_t = n_t_per[i]
            for k in range(K):
                centers_kl[k][i] = centers_concat[k, offset : offset + n_t].copy()
            offset += n_t

        # 4) 拟合 (a, b)
        params_mu_np = np.zeros((K, L, 2), dtype=np.float64)
        for k in range(K):
            for i in range(L):
                t = np.asarray(self.times_list[i], dtype=np.float64)
                y_center = centers_kl[k][i]
                a, b = fit_power_loglinear(
                    t,
                    y_center,
                    clip_a=(1e-8, np.inf),
                    clip_b=(-10.0, 10.0),
                )
                params_mu_np[k, i, 0] = a
                params_mu_np[k, i, 1] = b

        # 5) 估计 (phi, gamma)
        params_cov_np = np.zeros((K, L, 2), dtype=np.float64)
        for k in range(K):
            mask_k = labels_np == k
            n_k = int(mask_k.sum())
            for i in range(L):
                n_t = n_t_per[i]
                if n_k == 0:
                    params_cov_np[k, i] = (0.0, 1.0)
                    continue

                X_i_np = X_list[i].detach().cpu().numpy()
                X_ki = X_i_np[mask_k, :]  # (n_k, n_t)

                a = float(params_mu_np[k, i, 0])
                b = float(params_mu_np[k, i, 1])
                t = np.asarray(self.times_list[i], dtype=np.float64)
                with np.errstate(over="ignore", invalid="ignore"):
                    mu_curve = a * np.power(t, b)
                R = X_ki - mu_curve[None, :]                # (n_k, n_t)
                R = np.where(np.isfinite(R), R, 0.0)
                r_bar = R.mean(axis=0)                       # (n_t,)

                if n_t > 1 and np.std(r_bar) > 0:
                    with np.errstate(invalid="ignore"):
                        cc = np.corrcoef(r_bar[:-1], r_bar[1:])[0, 1]
                    if not np.isfinite(cc):
                        cc = 0.0
                    phi = float(np.clip(cc, -0.99, 0.99))
                    gamma_sq = max(float(np.var(r_bar)) * (1.0 - phi * phi), 1e-6)
                    gamma = float(np.sqrt(gamma_sq))
                elif n_t > 0:
                    phi = 0.0
                    gamma = float(np.sqrt(max(float(np.var(r_bar)), 1e-6)))
                else:
                    phi, gamma = 0.0, 1.0

                params_cov_np[k, i, 0] = phi
                params_cov_np[k, i, 1] = gamma

        # 6) numpy → torch & 写入 self
        labels_t = torch.from_numpy(labels_np).to(self.device).long()
        weights_t = torch.from_numpy(weights_np.astype(np.float64)).to(self.device, self.dtype)
        params_mu_t = torch.from_numpy(params_mu_np).to(self.device, self.dtype)
        params_cov_t = torch.from_numpy(params_cov_np).to(self.device, self.dtype)

        self.labels = labels_t
        self.weights = weights_t
        self.params_mu = params_mu_t
        self.params_cov = params_cov_t
        self._kmeans_init_backend = backend

        return {
            "labels": labels_t,
            "weights": weights_t,
            "mu_params": params_mu_t,
            "cov_params": params_cov_t,
            "centers_kl": centers_kl,
            "backend": backend,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Step 3: SAD1 covariance utilities
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _sad1_quad_per_feature(
        diff: torch.Tensor,
        phi: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """逐特征计算 SAD1 协方差下的二次型 ``(x - μ)ᵀ Σ⁻¹ (x - μ)``。

        利用 SAD1 协方差逆阵的三对角结构（``diag_inner = (1+φ²)/γ²``、
        ``diag_edge = 1/γ²``、``off = -φ/γ²``），对形如 ``(N, n_t)`` 的残差矩阵
        逐行直接求二次型，不显式构造 ``Σ⁻¹``。

        Args:
            diff: ``(N, n_t)`` 的残差张量 ``x - μ``。
            phi: 标量 0 维张量，AR(1) 系数；调用方应已夹紧到 ``(-0.995, 0.995)``。
            gamma: 标量 0 维张量，正数；调用方应已夹紧到 ``≥ 1e-6``。

        Returns:
            ``(N,)`` 的二次型张量。``n_t == 1`` 时蜕化为 ``diff² / γ²``。
        """
        n_t = int(diff.shape[-1])
        gamma_sq = gamma * gamma
        diag_inner = (1.0 + phi * phi) / gamma_sq
        diag_edge = 1.0 / gamma_sq
        off_diag = -phi / gamma_sq

        q = diag_inner * (diff * diff).sum(dim=-1)
        if n_t >= 1:
            edge_correction = (diag_edge - diag_inner) * (
                diff[..., 0] ** 2 + diff[..., -1] ** 2
            )
            q = q + edge_correction
        if n_t > 1:
            q = q + 2.0 * off_diag * (diff[..., :-1] * diff[..., 1:]).sum(dim=-1)
        return q

    @staticmethod
    def _sad1_logdet(phi: torch.Tensor, gamma: torch.Tensor, n_t: int) -> torch.Tensor:
        """SAD1 协方差矩阵的 log-determinant（闭式）。

        ``log|Σ| = 2·n·log γ - (n - 1)·log(1 - φ²)``。

        Args:
            phi: 标量张量，已夹紧到 ``(-0.995, 0.995)``。
            gamma: 标量张量，已夹紧到 ``≥ 1e-6``。
            n_t: 时间点数 ``n``。

        Returns:
            标量 0 维张量。
        """
        return 2.0 * n_t * torch.log(gamma) - (n_t - 1) * torch.log(1.0 - phi * phi)

    def _sad1_inv_block(
        self,
        params_cov_one: torch.Tensor,
        n_t: int,
    ) -> torch.Tensor:
        """构造单个 condition 的 SAD1 协方差三对角逆阵 ``(n_t, n_t)``。

        与二次型形式一致（``diag_inner = (1+φ²)/γ²``、``diag_edge = 1/γ²``、
        ``off = -φ/γ²``），主要用于外部诊断/验证。EM 主循环本身只用
        :meth:`_sad1_quad_per_feature` + :meth:`_sad1_logdet`，不显式构造此矩阵。

        Args:
            params_cov_one: 形如 ``(2,)`` 的 ``[phi, gamma]``。
            n_t: 时间点数。

        Returns:
            ``(n_t, n_t)`` 的逆协方差矩阵张量。
        """
        phi = torch.clamp(params_cov_one[0], -0.995, 0.995)
        gamma = torch.clamp(params_cov_one[1], min=1e-6)
        gamma_sq = gamma * gamma
        diag_inner = float((1.0 + phi * phi) / gamma_sq)
        diag_edge = float(1.0 / gamma_sq)
        off_diag = float(-phi / gamma_sq)

        device = params_cov_one.device
        dtype = params_cov_one.dtype
        inv_sigma = torch.zeros((n_t, n_t), device=device, dtype=dtype)
        diagonal = torch.full((n_t,), diag_inner, device=device, dtype=dtype)
        if n_t >= 1:
            diagonal[0] = diag_edge
            diagonal[-1] = diag_edge
        inv_sigma.diagonal().copy_(diagonal)
        if n_t > 1:
            inv_sigma.diagonal(1).fill_(off_diag)
            inv_sigma.diagonal(-1).fill_(off_diag)
        return inv_sigma

    # ────────────────────────────────────────────────────────────────────────
    # Step 4: E-step
    # ────────────────────────────────────────────────────────────────────────
    def _e_step(
        self,
        X_list: List[torch.Tensor],
        params_mu: torch.Tensor,
        params_cov: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """E 步：计算每个特征对每个簇的对数联合概率与后验责任。

        各 condition 的协方差为 SAD1 块对角，因此对每个 ``(k, i)`` 独立累加
        ``log|Σ_ki|`` 与 Mahalanobis ``(x-μ)ᵀ Σ⁻¹ (x-μ)``，再合成总维度
        ``d = Σ_i n_t_i`` 的多元正态对数密度。``log f_k`` 与 ``log π_k`` 相加后用
        ``logsumexp`` 得到样本对数似然，避免上溢/下溢。

        Args:
            X_list: 长度 ``L`` 的张量列表，第 i 项形如 ``(N, n_t_i)``。
            params_mu: ``(K, L, 2)``，最后一维为 ``[a, b]``。
            params_cov: ``(K, L, 2)``，最后一维为 ``[phi, gamma]``。
            weights: ``(K,)``，混合权重，未归一化时由内部 clamp + log 处理。

        Returns:
            ``(log_lik, resp)``：

            - ``log_lik``：标量 0 维张量，所有特征的对数似然之和 ``Σ_i log f_mix(x_i)``。
            - ``resp``：``(N, K)``，后验责任，行和归一化为 1。
        """
        if not X_list:
            raise ValueError("X_list 不能为空")
        if self.n_times_conditions is None or self.times_list is None:
            raise RuntimeError("_e_step 调用前必须先调用 _prepare_data")

        N = int(X_list[0].shape[0])
        K = self.n_components
        L = self.n_conditions
        n_t_per: List[int] = list(self.n_times_conditions)
        d_full = int(sum(n_t_per))

        log_probs = torch.zeros((N, K), device=self.device, dtype=self.dtype)
        const = -0.5 * d_full * float(np.log(2.0 * np.pi))

        for k in range(K):
            maha_total = torch.zeros(N, device=self.device, dtype=self.dtype)
            logdet_total = torch.zeros((), device=self.device, dtype=self.dtype)

            for i in range(L):
                X_i = X_list[i]
                n_t = n_t_per[i]
                t_i = torch.from_numpy(
                    np.asarray(self.times_list[i], dtype=np.float64)
                ).to(self.device, self.dtype)

                a = params_mu[k, i, 0]
                b = params_mu[k, i, 1]
                phi = torch.clamp(params_cov[k, i, 0], -0.995, 0.995)
                gamma = torch.clamp(params_cov[k, i, 1], min=1e-6)

                if not (
                    torch.isfinite(a)
                    and torch.isfinite(b)
                    and torch.isfinite(phi)
                    and torch.isfinite(gamma)
                ):
                    a = torch.tensor(1.0, device=self.device, dtype=self.dtype)
                    b = torch.tensor(0.5, device=self.device, dtype=self.dtype)
                    phi = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                    gamma = torch.tensor(1.0, device=self.device, dtype=self.dtype)

                mu_ki = a * (t_i ** b)
                diff = X_i - mu_ki.unsqueeze(0)

                q = self._sad1_quad_per_feature(diff, phi, gamma)
                q = torch.where(
                    torch.isfinite(q),
                    q,
                    torch.full_like(q, 1e6),
                )
                maha_total = maha_total + q
                logdet_total = logdet_total + self._sad1_logdet(phi, gamma, n_t)

            log_probs[:, k] = const - 0.5 * (logdet_total + maha_total)

        log_probs = torch.where(
            torch.isfinite(log_probs),
            log_probs,
            torch.full_like(log_probs, -1e10),
        )

        log_w = torch.log(torch.clamp(weights, min=1e-16))
        log_joint = log_probs + log_w.unsqueeze(0)
        log_norm = torch.logsumexp(log_joint, dim=1, keepdim=True)
        log_lik = log_norm.sum()
        resp = torch.exp(log_joint - log_norm)
        resp = torch.where(torch.isfinite(resp), resp, torch.full_like(resp, 1.0 / K))
        resp = resp / resp.sum(dim=1, keepdim=True).clamp(min=1e-16)
        return log_lik, resp

    # ────────────────────────────────────────────────────────────────────────
    # Step 5: M-step（方案 A：a 闭式 + b 冻；gamma 闭式 + phi 冻）
    # ────────────────────────────────────────────────────────────────────────
    def _m_step(
        self,
        X_list: List[torch.Tensor],
        resp: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """M 步：按方案 A 更新 ``weights / a / gamma``，``b / phi`` 沿用上一轮。

        - **weights**：``π_k = N_k / N``，``N_k = Σ_i resp[i, k]``，归一化和为 1。
        - **a_{k,i}**：以上一轮 ``b`` 与协方差对角主项简化为加权 OLS：

            ``a = Σ_i q_{ki} · X_i · t^b / Σ_i q_{ki} · t^{2b}``

          内层重复 3 次（保留接口便于以后改为同时更新 ``a, b``），外加 ``a ≥ 1e-8`` 兜底。
        - **gamma_{k,i}**：``γ² = Σ_i q_{ki} · ξ_i / (n_t · N_k)``，其中 ``ξ_i`` 为
          以 ``γ = 1`` 计算的 SAD1 二次型。``γ`` 必为正（取平方根，最小 ``1e-4``）。
        - **phi_{k,i}**：保持上一轮值；若该簇空簇或本步出现 NaN，则用残差自相关
          重估并夹紧 ``[-0.99, 0.99]``，此时 ``γ`` 也按相应方差闭式重设。

        Args:
            X_list: ``_prepare_data`` 输出的张量列表。
            resp: ``(N, K)`` 的后验责任。

        Returns:
            字典 ``{"mu_params", "cov_params", "weights"}``。
        """
        if self.params_mu is None or self.params_cov is None:
            raise RuntimeError("_m_step 调用前必须先有上一轮的 params_mu / params_cov")

        N = int(X_list[0].shape[0])
        K = self.n_components
        L = self.n_conditions
        n_t_per: List[int] = list(self.n_times_conditions or [])

        Nk = resp.sum(dim=0)
        weights = Nk / max(N, 1)
        weights = weights / weights.sum().clamp(min=1e-16)

        params_mu_new = self.params_mu.clone()
        params_cov_new = self.params_cov.clone()

        for k in range(K):
            if float(Nk[k]) < 1e-10:
                # 空簇：保留上一轮参数，跳过更新
                continue

            q_k = resp[:, k].detach()

            for i in range(L):
                X_i = X_list[i]
                n_t = n_t_per[i]
                t_i = torch.from_numpy(
                    np.asarray(self.times_list[i], dtype=np.float64)
                ).to(self.device, self.dtype)

                # b、phi 冻在上一轮
                b_prev = self.params_mu[k, i, 1].detach()
                phi_prev = self.params_cov[k, i, 0].detach()

                b = torch.clamp(b_prev, -10.0, 10.0)
                phi = torch.clamp(phi_prev, -0.995, 0.995)

                # 5.1 闭式更新 a（内层迭代 3 次，与参考实现一致）
                with torch.no_grad():
                    a_new = self.params_mu[k, i, 0].detach().clone()
                    for _ in range(3):
                        t_pow_b = t_i ** b
                        weighted = q_k.unsqueeze(1)
                        numer = (weighted * X_i * t_pow_b.unsqueeze(0)).sum()
                        denom = (weighted * (t_pow_b.unsqueeze(0) ** 2)).sum()
                        a_new = numer / denom.clamp(min=1e-8)
                        a_new = torch.clamp(a_new, min=1e-8)

                # 5.2 计算残差并以"phi 冻 + gamma=1"二次型估 gamma²
                with torch.no_grad():
                    mean_curve = a_new * (t_i ** b)
                    diff = X_i - mean_curve.unsqueeze(0)

                    one = torch.ones((), device=self.device, dtype=self.dtype)
                    q_per_feature = self._sad1_quad_per_feature(diff, phi, one)
                    gamma_sq = (q_k * q_per_feature).sum() / max(n_t, 1) / Nk[k]
                    gamma_sq = torch.clamp(gamma_sq, min=1e-8)
                    gamma_new = torch.sqrt(gamma_sq)
                    phi_new = phi

                    if not (torch.isfinite(gamma_new) and torch.isfinite(phi_new)):
                        # 兜底：用加权残差均值序列重估 (phi, gamma)
                        weighted_diff = diff * q_k.unsqueeze(1)
                        r_mean = weighted_diff.sum(dim=0) / Nk[k].clamp(min=1e-16)
                        if n_t > 1:
                            r_np = r_mean.detach().cpu().numpy()
                            with np.errstate(invalid="ignore"):
                                cc = np.corrcoef(r_np[:-1], r_np[1:])[0, 1]
                            cc = float(cc) if np.isfinite(cc) else 0.0
                            phi_alt = float(np.clip(cc, -0.99, 0.99))
                            var_alt = float(np.var(r_np)) * (1.0 - phi_alt ** 2)
                            phi_new = torch.tensor(
                                phi_alt, device=self.device, dtype=self.dtype
                            )
                            gamma_new = torch.tensor(
                                float(np.sqrt(max(var_alt, 1e-6))),
                                device=self.device,
                                dtype=self.dtype,
                            )
                        else:
                            phi_new = torch.tensor(
                                0.0, device=self.device, dtype=self.dtype
                            )
                            gamma_new = torch.tensor(
                                1.0, device=self.device, dtype=self.dtype
                            )

                    gamma_new = torch.clamp(gamma_new, min=1e-4)
                    phi_new = torch.clamp(phi_new, -0.995, 0.995)

                params_mu_new[k, i, 0] = a_new
                params_mu_new[k, i, 1] = b
                params_cov_new[k, i, 0] = phi_new
                params_cov_new[k, i, 1] = gamma_new

        return {
            "mu_params": params_mu_new,
            "cov_params": params_cov_new,
            "weights": weights,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Step 6: fit / predict / 辅助接口
    # ────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        data: List[pd.DataFrame],
        *,
        verbose: bool = False,
    ) -> "FunClu":
        """端到端拟合：``_prepare_data → _initialize → EM 主循环``。

        EM 终止条件（任一满足即停）：

        1. ``|Δlog_lik| < tol``（视为收敛，``self.converged = True``）；
        2. 达到 ``max_iter``；
        3. 出现 NaN/Inf log-likelihood 或参数（早停，回退到上一轮参数）；
        4. ``log_lik`` 显著下降（差值 ``> 1e6``，认为发散，早停）。

        终止后用最终参数再跑一次 E 步得到 ``resp``，``labels = argmax resp``；若
        发现空簇，回退为初始化的 KMeans 标签（保证每簇至少一个成员）。

        BIC 与 ``n_params`` 计算与 R 端一致：``BIC = 2·NLL + p·log(N)``，
        ``NLL = -log_likelihood``，``p = K · L · 4 + max(K - 1, 0)``。

        Args:
            data: 多 condition 的 DataFrame 列表（与 ``_prepare_data`` 同形）。
            verbose: 是否在终端打印每轮 log-likelihood / 收敛信息（不调用 ``st.*``）。

        Returns:
            ``self``，便于链式调用。
        """
        X_list, _ = self._prepare_data(data)
        init_result = self._initialize(X_list)
        self.params_mu = init_result["mu_params"]
        self.params_cov = init_result["cov_params"]
        self.weights = init_result["weights"]
        init_labels: torch.Tensor = init_result["labels"].clone()

        N = int(X_list[0].shape[0])
        prev_log_likelihood = -float("inf")
        last_log_likelihood = -float("inf")
        self.converged = False
        self.loglik_history = []
        self.n_iter_run = 0

        if verbose:
            print(
                f"[FunClu] EM start | N={N} | K={self.n_components} | "
                f"L={self.n_conditions} | n_times={self.n_times_conditions} | "
                f"max_iter={self.max_iter} | tol={self.tol}",
                flush=True,
            )

        for iter_num in range(self.max_iter):
            log_lik_t, resp = self._e_step(
                X_list, self.params_mu, self.params_cov, self.weights
            )
            log_likelihood = float(log_lik_t.item())

            if not np.isfinite(log_likelihood):
                if verbose:
                    print(
                        f"[FunClu] iter {iter_num + 1}: NaN/Inf log-lik，提前停止。",
                        flush=True,
                    )
                log_likelihood = prev_log_likelihood
                break

            self.loglik_history.append(log_likelihood)
            self.n_iter_run = iter_num + 1
            last_log_likelihood = log_likelihood

            m_result = self._m_step(X_list, resp)
            if (
                torch.isnan(m_result["mu_params"]).any()
                or torch.isnan(m_result["cov_params"]).any()
            ):
                if verbose:
                    print(
                        f"[FunClu] iter {iter_num + 1}: NaN in M-step params，提前停止。",
                        flush=True,
                    )
                break

            self.params_mu = m_result["mu_params"]
            self.params_cov = m_result["cov_params"]
            self.weights = m_result["weights"]

            if verbose:
                print(
                    f"[FunClu] iter {iter_num + 1}/{self.max_iter} "
                    f"log-lik={log_likelihood:.6f}",
                    flush=True,
                )

            change = abs(log_likelihood - prev_log_likelihood)
            if np.isfinite(prev_log_likelihood) and change < self.tol:
                if verbose:
                    print(
                        f"[FunClu] converged at iter {iter_num + 1} "
                        f"(Δ={change:.2e} < tol={self.tol:.2e})",
                        flush=True,
                    )
                self.converged = True
                break

            if (
                np.isfinite(prev_log_likelihood)
                and log_likelihood < prev_log_likelihood - 1e6
            ):
                if verbose:
                    print(
                        f"[FunClu] iter {iter_num + 1}: log-lik 显著下降，提前停止。",
                        flush=True,
                    )
                break

            prev_log_likelihood = log_likelihood

        # 用最终参数再跑一次 E 步，分配标签
        _, final_resp = self._e_step(
            X_list, self.params_mu, self.params_cov, self.weights
        )
        final_labels = final_resp.argmax(dim=1)
        counts_final = torch.bincount(final_labels, minlength=self.n_components)
        if (counts_final == 0).any():
            if verbose:
                empty_ids = [i for i, c in enumerate(counts_final.tolist()) if c == 0]
                print(
                    f"[FunClu] empty cluster(s) {empty_ids} after EM；"
                    f"回退到 KMeans 初始化标签。",
                    flush=True,
                )
            self.labels = init_labels
        else:
            self.labels = final_labels

        ell = (
            last_log_likelihood
            if np.isfinite(last_log_likelihood)
            else (prev_log_likelihood if np.isfinite(prev_log_likelihood) else float("nan"))
        )
        self.log_likelihood = ell
        self.neg_log_likelihood = -ell if np.isfinite(ell) else float("nan")
        self.n_params = (
            self.n_components * self.n_conditions * 4
            + max(self.n_components - 1, 0)
        )
        if np.isfinite(ell):
            self.bic = 2.0 * (-ell) + self.n_params * float(np.log(max(N, 1)))
        else:
            self.bic = float("nan")

        if verbose:
            print(
                f"[FunClu] EM end | converged={self.converged} | "
                f"log-lik={self.log_likelihood} | BIC={self.bic}",
                flush=True,
            )
        return self

    def predict(self, data: List[pd.DataFrame]) -> torch.Tensor:
        """对新数据按当前模型分配硬标签 ``argmax resp``。

        Args:
            data: 多 condition 的 DataFrame 列表，列必须能与训练阶段的 ``common_cols``
                取交集（少于训练时会重新缩窄 ``common_cols``，由 ``_prepare_data`` 处理）。

        Returns:
            ``(N,)`` 的 ``torch.LongTensor``。

        Raises:
            RuntimeError: 模型尚未拟合（``params_mu`` 为 ``None``）。
        """
        if self.params_mu is None or self.params_cov is None or self.weights is None:
            raise RuntimeError("predict 调用前必须先调用 fit / _initialize")
        X_list, _ = self._prepare_data(data)
        _, resp = self._e_step(X_list, self.params_mu, self.params_cov, self.weights)
        return resp.argmax(dim=1)

    def get_params(self) -> Dict[str, Optional[np.ndarray]]:
        """以 numpy 字典导出关键模型参数，便于持久化或表格展示。

        Returns:
            字典：``mu_params (K,L,2)``、``cov_params (K,L,2)``、``weights (K,)``、
            ``labels (N,)``；尚未填充的字段为 ``None``。
        """
        return {
            "mu_params": (
                self.params_mu.detach().cpu().numpy()
                if self.params_mu is not None
                else None
            ),
            "cov_params": (
                self.params_cov.detach().cpu().numpy()
                if self.params_cov is not None
                else None
            ),
            "weights": (
                self.weights.detach().cpu().numpy()
                if self.weights is not None
                else None
            ),
            "labels": (
                self.labels.detach().cpu().numpy()
                if self.labels is not None
                else None
            ),
        }

    def get_cluster_curves(
        self,
        condition_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """在指定 condition 下，按当前 ``params_mu`` 计算各簇的均值曲线 ``a · t^b``。

        Args:
            condition_idx: condition 索引，范围 ``[0, n_conditions)``。

        Returns:
            ``(times, curves)``：``times`` 形如 ``(n_t_i,)``，``curves`` 形如
            ``(n_components, n_t_i)``。

        Raises:
            RuntimeError: 模型尚未填充 ``params_mu`` / ``times_list``。
            IndexError: ``condition_idx`` 越界。
        """
        if self.params_mu is None or self.times_list is None:
            raise RuntimeError("get_cluster_curves 调用前必须先调用 fit / _initialize")
        if not (0 <= condition_idx < self.n_conditions):
            raise IndexError(
                f"condition_idx={condition_idx} 越界（n_conditions={self.n_conditions}）"
            )
        t = np.asarray(self.times_list[condition_idx], dtype=np.float64)
        mu_np = self.params_mu.detach().cpu().numpy()
        curves = np.empty((self.n_components, t.size), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for k in range(self.n_components):
                a = float(mu_np[k, condition_idx, 0])
                b = float(mu_np[k, condition_idx, 1])
                curves[k] = a * np.power(t, b)
        return t, curves

    def __repr__(self) -> str:
        return (
            f"FunClu(n_components={self.n_components}, "
            f"n_conditions={self.n_conditions}, "
            f"n_features={self.n_features}, "
            f"backend={self._kmeans_init_backend!r})"
        )


def compute_bic_scores(
    data: List[pd.DataFrame],
    k_min: int = 2,
    k_max: int = 10,
    step: int = 1,
    *,
    max_iter: int = 50,
    tol: float = 1e-4,
    random_state: int = 42,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Scan a range of K values and return BIC scores.

    For each K in [k_min, k_max] with given step, instantiates a
    :class:`FunClu`, calls :meth:`~FunClu.fit`, and collects BIC, log-likelihood,
    convergence status, etc.  Single-K failures are recorded as NaN rows rather
    than aborting the scan.

    Args:
        data: One DataFrame per condition (same format as :meth:`FunClu.fit`).
        k_min: Smallest K to try (must be ≥ 2).
        k_max: Largest K to try (capped to the number of features in *data*).
        step: Stride between consecutive K values.
        max_iter: Passed to :class:`FunClu`.
        tol: Passed to :class:`FunClu`.
        random_state: Passed to :class:`FunClu`.
        verbose: Passed to :meth:`FunClu.fit`.
        progress_callback: Called after each K as ``callback(idx_1based, total)``.

    Returns:
        DataFrame with columns ``K``, ``BIC``, ``log_likelihood``, ``NLL``,
        ``n_params``, ``converged``, ``n_iter_run``, ``n_features``,
        ``n_conditions``.
    """
    if k_min < 2:
        raise ValueError(f"k_min must be >= 2, got {k_min}")
    if k_max < k_min:
        raise ValueError(f"k_max ({k_max}) must be >= k_min ({k_min})")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    n_features = min((d.shape[1] for d in data), default=0)
    if n_features < 2:
        raise ValueError(f"Data has fewer than 2 features (got {n_features})")
    capped_k_max = min(k_max, n_features)

    ks = list(range(k_min, capped_k_max + 1, step))
    if not ks:
        raise ValueError(
            f"No K values to scan: k_min={k_min}, "
            f"k_max={k_max}, n_features={n_features}"
        )

    rows: List[Dict[str, Any]] = []
    total = len(ks)

    for idx, k in enumerate(ks):
        row: Dict[str, Any] = {
            "K": k, "BIC": float("nan"), "log_likelihood": float("nan"),
            "NLL": float("nan"), "n_params": 0, "converged": False,
            "n_iter_run": 0, "n_features": 0, "n_conditions": 0,
        }
        try:
            model = FunClu(
                n_components=k, max_iter=max_iter, tol=tol,
                random_state=random_state,
            )
            model.fit(data, verbose=verbose)
            row.update({
                "BIC": model.bic,
                "log_likelihood": model.log_likelihood,
                "NLL": model.neg_log_likelihood,
                "n_params": model.n_params,
                "converged": model.converged,
                "n_iter_run": model.n_iter_run,
                "n_features": model.n_features,
                "n_conditions": model.n_conditions,
            })
        except Exception:
            pass
        rows.append(row)

        if progress_callback is not None:
            progress_callback(idx + 1, total)

    return pd.DataFrame(rows)
