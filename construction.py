"""Network Construction 后端计算模块。"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
from scipy.special import eval_legendre, legendre

MIN_ASGL_MAX_ORDER = 5


def polynomial_basis_expansion(
    data: pd.DataFrame,
    max_order: int,
) -> pd.DataFrame:
    """对每个特征做 Legendre 基展开，第 ``(k, r)`` 列为 ``y_k(τ) · P_r(τ̂)``。

    - 阶数 ``r`` 从 0 起步，取 ``r = 0, 1, …, max_order - 1``；每个变量共
      ``max_order`` 列。``P_0 ≡ 1``，故 r=0 列即 ``y_k(τ)`` 本身。
    - τ 取自 ``data.index``（视作时间轴）；内部线性归一化到
      ``τ̂ ∈ [-1, 1]``（``τ_min → -1``、``τ_max → +1``），再喂给 Legendre 多项式；
      若 ``τ_max == τ_min``，则 ``τ̂`` 取 0。
    - y_k(τ) 为 ``data`` 的第 k 列在对应行的取值。

    返回 DataFrame 列顺序为
    ``[(k=0, r=0), …, (k=0, r=max_order-1), (k=1, r=0), …]``，
    列名 ``f"{feature}_o({r})"``，便于下游按 ``k*max_order + r`` 分组。
    """
    values = data.values.astype(float)
    n_samples, n_features = values.shape

    tau = np.asarray(data.index, dtype=float)
    tau_min = float(tau.min())
    tau_max = float(tau.max())
    if tau_max > tau_min:
        tau_hat = -1.0 + 2.0 * (tau - tau_min) / (tau_max - tau_min)
    else:
        tau_hat = np.zeros_like(tau)

    per_order_tau = [eval_legendre(order, tau_hat) for order in range(max_order)]
    basis_tau = np.stack(per_order_tau, axis=0)
    basis_arr = basis_tau[:, :, np.newaxis] * values[np.newaxis, :, :]
    basis_arr = basis_arr.transpose(1, 2, 0)

    columns = [
        f"{data.columns[i]}_o({order})"
        for i in range(n_features)
        for order in range(max_order)
    ]
    flat = basis_arr.reshape(n_samples, n_features * max_order)
    return pd.DataFrame(flat, index=data.index, columns=columns)


def tiger_lop_basis_expansion(
    data: pd.DataFrame,
    max_order: int,
) -> pd.DataFrame:
    """构造 TIGER/LOP 风格的逐源 Legendre 导数积分基。

    对每个 source 曲线 ``x_k(t)``，先把它自身缩放到 ``[-1, 1]``，再把
    ``P_r(x_k)`` 沿 ``x_k`` 的缩放坐标做梯形积分。每个 source 的所有基列首点均为
    0，因此 self effect 可由目标初值单独承载，cross effect 天然从 0 起步。
    """
    values = data.values.astype(float)
    n_samples, n_features = values.shape
    out = np.zeros((n_samples, n_features * max_order), dtype=float)
    columns: list[str] = []

    for k, feature_name in enumerate(data.columns):
        x = values[:, k]
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max > x_min:
            x_scaled = -1.0 + 2.0 * (x - x_min) / (x_max - x_min)
        else:
            x_scaled = np.zeros_like(x)
        dx = np.diff(x_scaled)
        for order in range(max_order):
            deriv_values = eval_legendre(order, x_scaled)
            integral = np.zeros(n_samples, dtype=float)
            if n_samples > 1:
                increments = 0.5 * (
                    deriv_values[1:] + deriv_values[:-1]
                ) * dx
                integral[1:] = np.cumsum(increments)
            out[:, k * max_order + order] = integral
            columns.append(f"{feature_name}_o({order})")

    return pd.DataFrame(out, index=data.index, columns=columns)


def polynomial_basis_expansion_integral_analytic(
    data: pd.DataFrame,
    power_function_params: pd.DataFrame,
    max_order: int,
) -> pd.DataFrame:
    """解析地构造 integral 模式下的设计矩阵列。

    在 ``y_k(τ) = a_k · τ^{b_k}`` 的解析假设下（与 :func:`curve_fitting
    .get_power_function_sample` 一致），对积分

    .. math::

        \\Phi^{\\text{int}}_{k,r}(\\tau) = \\int_{\\tau_1}^{\\tau}
            a_k\\, s^{b_k}\\, P_r\\!\\bigl(\\hat\\tau(s)\\bigr)\\,ds

    在每个采样点解析求值，消除 ``cumulative_trapezoid`` 的梯形误差。其中

    - ``τ_1 = data.index.min()`` 为积分下限；
    - ``τ̂(s) = α s + β``，``α = 2 / (τ_max - τ_min)``，``β = -1 - α τ_min``；
    - ``P_r`` 为 Legendre 多项式（``scipy.special.legendre``）；
    - ``a_k, b_k`` 来自 ``power_function_params`` 的 ``"a", "b"`` 两行
      （列与 ``data.columns`` 通过 ``reindex`` 对齐）。

    推导：把 ``P_r(x) = Σ_m c_{r,m} x^m`` 与 ``x = α s + β`` 代入并展开后

    .. math::

        y_k(s)\\, P_r(\\hat\\tau(s)) = a_k \\sum_{j=0}^{r} d_{r,j}\\, s^{b_k+j},
        \\quad d_{r,j} = \\sum_{m=j}^{r} c_{r,m} \\binom{m}{j} \\alpha^j \\beta^{m-j}.

    逐项原函数为 ``s^{b_k+j+1} / (b_k+j+1)``（典型 ``b_k > 0`` 下分母非零；
    若 ``b_k+j+1 == 0`` 则单独走 ``ln`` 分支以保稳）。

    返回 DataFrame 列顺序与 :func:`polynomial_basis_expansion` 一致，便于
    下游按 ``k*max_order + r`` 分组复用。
    """
    tau = np.asarray(data.index, dtype=float)
    n_samples = tau.shape[0]
    feature_names = list(data.columns)
    n_features = len(feature_names)

    params = power_function_params.reindex(feature_names)
    if params.isna().to_numpy().any():
        missing = params.index[params.isna().any(axis=1)].tolist()
        raise ValueError(
            "power_function_params 与曲线列不对齐，缺失列 a/b："
            f"{missing}"
        )
    a_arr = params["a"].to_numpy(dtype=float)
    b_arr = params["b"].to_numpy(dtype=float)

    tau_min = float(tau.min())
    tau_max = float(tau.max())
    tau1 = tau_min
    if tau_max > tau_min:
        alpha = 2.0 / (tau_max - tau_min)
        beta = -1.0 - alpha * tau_min
    else:
        alpha, beta = 0.0, 0.0

    out = np.zeros((n_samples, n_features * max_order), dtype=float)
    for r in range(max_order):
        cr = legendre(r).coef[::-1]
        d_rj = np.zeros(r + 1, dtype=float)
        for j in range(r + 1):
            acc = 0.0
            for m in range(j, r + 1):
                acc += cr[m] * comb(m, j) * (alpha ** j) * (beta ** (m - j))
            d_rj[j] = acc
        for k in range(n_features):
            col_idx = k * max_order + r
            col_vals = np.zeros(n_samples, dtype=float)
            for j in range(r + 1):
                exp = b_arr[k] + j + 1
                coef = a_arr[k] * d_rj[j]
                if coef == 0.0:
                    continue
                if abs(exp) < 1e-12:
                    safe_tau1 = tau1 if tau1 > 0 else np.finfo(float).tiny
                    safe_tau = np.where(tau > 0, tau, np.finfo(float).tiny)
                    col_vals += coef * (np.log(safe_tau) - np.log(safe_tau1))
                else:
                    col_vals += coef * (
                        np.power(tau, exp) - np.power(tau1, exp)
                    ) / exp
            out[:, col_idx] = col_vals

    columns = [
        f"{feature_names[k]}_o({r})"
        for k in range(n_features)
        for r in range(max_order)
    ]
    return pd.DataFrame(out, index=data.index, columns=columns)


def align_response_to_design(
    response_df: pd.DataFrame,
    target_index: pd.Index | np.ndarray,
) -> pd.DataFrame:
    """将响应数据（如 quasi_dynamic_df）按数值索引线性插值到 ``target_index``。

    用于设计矩阵 ``X`` 在切比雪夫节点上、响应数据在原始观测点上时的对齐：

    - 源索引（``response_df.index``）与目标索引一般不重合，直接 ``reindex``
      会全部得到 NaN；本函数按数值大小对齐，用 :func:`numpy.interp` 逐列做
      线性插值。
    - 内部先把源数据按 index 升序整理；目标索引中超出源范围的点取最近端点
      （``np.interp`` 的默认外推行为，等价于 ``fill_value=(y[0], y[-1])``）。
    - 列名 / 列顺序保持不变；返回 DataFrame 的 index 为 ``target_index``。
    - 若 ``response_df.index`` 与 ``target_index`` 数值上完全相同（含顺序），
      插值结果就是原数据；因此对"已经在切比雪夫节点上"的输入是 no-op。
    """
    src_idx = np.asarray(response_df.index, dtype=float)
    tgt_idx = np.asarray(target_index, dtype=float)
    order = np.argsort(src_idx, kind="stable")
    src_sorted = src_idx[order]
    values_sorted = response_df.values.astype(float)[order]
    n_targets = values_sorted.shape[1]
    out = np.empty((len(tgt_idx), n_targets), dtype=float)
    for j in range(n_targets):
        out[:, j] = np.interp(tgt_idx, src_sorted, values_sorted[:, j])
    return pd.DataFrame(out, index=pd.Index(tgt_idx), columns=response_df.columns)


def _apply_column_normalization(
    Xv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对设计矩阵的非截距列做"纯 L2 单位化"（不中心化）。

    - ``Xv`` 第 0 列假定为截距列（全 1），不动；其余列每列除以列 L2 范数，使每列在
      归一化空间下 L2 长度为 1，但列均值保持不变（不再做中心化）。
    - 返回 ``(Xv_scaled, mu, sigma)``：

      * ``Xv_scaled[:, 0] == Xv[:, 0]``（截距列原样）；
      * ``mu`` 始终全 0（仅作占位以兼容 :func:`_denormalize_coefficients` 的统一接口）；
      * ``sigma[0] = 1`` 占位；
      * ``sigma[c]`` = 原始第 c 列的 L2 范数；若 ``< 1e-12``（列近似为 0）则置 1，不缩放。
    - 该归一化是纯列缩放：``Phi_scaled = Phi / sigma``。归一化空间下"固定截距 = y0"的
      OLS / ASGL 与原始空间下"固定截距 = y0"的 OLS / ASGL **严格等价**（即
      ``theta_orig = theta_scaled / sigma``、``W[0, j] = W_scaled[0, j]``），从而保留
      Effect Decomposition 的真实语义：``predict_j = W[0, j] + sum_k effect_{k->j}``，
      ``W[0, j] = y0[j]``、effects 等价于原始空间下的固定截距解。
    """
    p = Xv.shape[1]
    Xv_scaled = Xv.astype(float, copy=True)
    mu = np.zeros(p, dtype=float)
    sigma = np.ones(p, dtype=float)
    if p <= 1:
        return Xv_scaled, mu, sigma
    body = Xv_scaled[:, 1:]
    norms = np.linalg.norm(body, axis=0)
    sigma[1:] = np.where(norms > 1e-12, norms, 1.0)
    Xv_scaled[:, 1:] = body / sigma[1:]
    return Xv_scaled, mu, sigma


def _denormalize_coefficients(
    W_scaled: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """把归一化空间下的系数矩阵反算回原始空间。

    给定 ``X_scaled = [1 | (Φ - μ) / σ]``、``W_scaled = [β̃; θ̃]``，反算后

    - ``θ_orig = θ̃ / σ``
    - ``β_orig = β̃ - μ_body^⊤ · θ_orig``

    使得 ``X_orig @ W_orig == X_scaled @ W_scaled``（仿射等价）。

    与 :func:`_apply_column_normalization` 配合的当前实现里 ``μ ≡ 0``（纯 L2 缩放、
    不中心化），公式自然退化为 ``β_orig = β̃``、``θ_orig = θ̃ / σ``。保留 ``μ``
    入参是为了在未来切换归一化策略时仍可复用同一接口。
    """
    W = np.zeros_like(W_scaled, dtype=float)
    W[1:, :] = W_scaled[1:, :] / sigma[1:, np.newaxis]
    W[0, :] = W_scaled[0, :] - mu[1:] @ W[1:, :]
    return W


def _project_nonneg_self_keep_intercept(
    w: np.ndarray,
    Xv: np.ndarray,
    intercept_idx: int,
    self_group: np.ndarray,
    pinv: np.ndarray | None = None,
    allow_intercept_relax: bool = False,
) -> None:
    """投影使自效应 ``β + Φ_self @ θ_self >= 0`` 处处成立。

    默认行为：**严格不动截距**——通过最小二乘把 ``θ_self`` 投影到使整体非负的目标上；
    与 ``intercept_values`` 显式锁定的语义一致。

    当 ``allow_intercept_relax=True`` 时（用于释放截距路径）：投影后若 ``self_group``
    列张成空间不足以覆盖目标（残差仍含负值），再把截距上抬恰好等于负残差幅度作为兜底，
    避免 ``θ_self`` 被拉成数量级远高于原解的失真值。残差可忽略时截距不动，行为与默认一致。
    """
    e = Xv[:, self_group] @ w[self_group] + w[intercept_idx]
    if float(e.min()) >= 0:
        return
    e_clamped = np.maximum(e, 0.0)
    e_self_target = e_clamped - w[intercept_idx]
    if pinv is not None:
        theta_new = pinv @ e_self_target
    else:
        theta_new = np.linalg.lstsq(
            Xv[:, self_group], e_self_target, rcond=None
        )[0]
    w[self_group] = theta_new
    if not allow_intercept_relax:
        return
    e_new = Xv[:, self_group] @ theta_new + w[intercept_idx]
    gap = float(e_new.min())
    if gap < -1e-9:
        w[intercept_idx] += -gap


def _precompute_pinvs(Xv: np.ndarray, groups: list[np.ndarray]) -> dict[int, np.ndarray]:
    """预计算每个非截距组的伪逆，供约束投影复用。"""
    pinvs: dict[int, np.ndarray] = {}
    for gi, g in enumerate(groups):
        if gi == 0:
            continue
        pinvs[gi] = np.linalg.pinv(Xv[:, g])
    return pinvs


def _asgl_col(
    Xv: np.ndarray,
    y: np.ndarray,
    groups: list[np.ndarray],
    lam: float,
    mix: float,
    coef_weights: np.ndarray,
    group_weights: np.ndarray,
    protected_ids: set[int] | None = None,
    target_idx: int = -1,
    nonneg_self: bool = False,
    max_iter: int = 2000,
    tol: float = 1e-5,
    w_init: np.ndarray | None = None,
    pinvs: dict[int, np.ndarray] | None = None,
    XtX: np.ndarray | None = None,
    Xty: np.ndarray | None = None,
    L: float | None = None,
    fixed_intercept: float | None = None,
) -> np.ndarray:
    """Adaptive Sparse Group Lasso (ASGL) 近端梯度下降。"""
    if protected_ids is None:
        protected_ids = set()
    n, p = Xv.shape
    if XtX is None:
        XtX = Xv.T @ Xv
    if Xty is None:
        Xty = Xv.T @ y
    if L is None:
        L = float(np.linalg.norm(XtX, ord=2))
    if L == 0:
        return np.zeros(p)
    step = 1.0 / L

    self_pinv = pinvs.get(target_idx) if pinvs else None
    w = w_init.copy() if w_init is not None else np.zeros(p)
    if fixed_intercept is not None:
        w[0] = fixed_intercept

    for it in range(max_iter):
        w_prev = w.copy()
        grad = XtX @ w - Xty
        u = w - step * grad
        if fixed_intercept is not None:
            u[0] = fixed_intercept

        for gi, g in enumerate(groups):
            if len(g) == 0:
                continue
            if gi == 0 or gi in protected_ids:
                w[g] = u[g]
                continue
            ug = u[g].copy()
            if mix > 0:
                thresh_l1 = lam * mix * coef_weights[g] * step
                ug = np.sign(ug) * np.maximum(np.abs(ug) - thresh_l1, 0.0)
            norm_ug = np.linalg.norm(ug)
            if norm_ug == 0.0:
                w[g] = 0.0
            elif mix < 1:
                pg = float(len(g))
                thresh_grp = (
                    lam * (1.0 - mix) * float(group_weights[gi]) * np.sqrt(pg) * step
                )
                if norm_ug > thresh_grp:
                    w[g] = (1.0 - thresh_grp / norm_ug) * ug
                else:
                    w[g] = 0.0
            else:
                w[g] = ug

        if (it % 10 == 0 or it < 10) and nonneg_self and target_idx > 0:
            _project_nonneg_self_keep_intercept(
                w,
                Xv,
                0,
                groups[target_idx],
                pinv=self_pinv,
                allow_intercept_relax=fixed_intercept is None,
            )

        if np.linalg.norm(w - w_prev) < tol:
            break

    if nonneg_self and target_idx > 0:
        _project_nonneg_self_keep_intercept(
            w,
            Xv,
            0,
            groups[target_idx],
            pinv=self_pinv,
            allow_intercept_relax=fixed_intercept is None,
        )
    return w


def _build_groups(n_features: int, max_order: int) -> list[np.ndarray]:
    """构建分组列表：[截距组] + [每个源变量的 max_order 个基函数列]。"""
    groups: list[np.ndarray] = [np.array([0])]
    for k in range(n_features):
        start = 1 + k * max_order
        groups.append(np.arange(start, start + max_order))
    return groups


def _range(values: np.ndarray) -> float:
    """返回一维数组的有限极差。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite) - np.min(finite))


class IDOPRegressor:
    """多输出线性回归：单阶段 self-dominant ASGL（self/cross 区分惩罚权重）。

    求解结构（``fit`` → ``_fit_asgl_bic``）：

    - **单阶段联合 ASGL**：对每个目标 ``j`` 在完整设计矩阵 ``[intercept | Φ]``
      上跑 Adaptive Sparse Group Lasso；self 组（``gi == j + 1``）与 cross 组
      （``gi != 0`` 且 ``gi != j + 1``）共享同一个目标函数，但 cross 组的
      group lasso / L1 惩罚被**乘以 ``cross_weight_ratio``**（默认 5），使 cross
      在 ASGL 选择中"更贵"，self 在相同 ``alpha`` 下更优先被激活并保留较大幅值。
    - BIC 两层网格：外层 ``max_order``（``[MIN_ASGL_MAX_ORDER, user_max_order]``），
      内层 ``mix``（``[0, .25, .5, .75, 1]`` 或固定 ``self.mix``）× ``alpha``。
    - **同步增长数据**：BIC 选 ``alpha`` 偏大 → cross 几乎全部为 0、self 100% 主导，
      如实反映"无跨变量信号"。
    - **有 cross 信号的数据**：BIC 选 ``alpha`` 适中 → cross 部分激活
      （被压小、可正可负），self 仍主导。

    其他机制：``nonneg_self`` 在每次迭代和收敛后投影 self group 使自效应非负;
    ``max_interactions`` 对最终系数做 Top-K 跨变量保留 +
    ``[self_g | keep_cross_g]`` 一次合并 OLS 收尾（并复 nonneg_self 投影）;
    ``adaptive_weights`` 控制 ASGL 的自适应权重（默认 ``False``——纯结构权重，
    不被 OLS 信号偏置）; ``basis_decay`` 仍作用在 ``_design`` 的列尺度上。
    """

    def __init__(
        self,
        max_order: int,
        alpha: float = 1.0,
        mix: float = 0.5,
        fix_mix: bool = False,
        nonneg_self: bool = True,
        basis_decay: float = 0.0,
        max_interactions: int = 0,
        basis_type: str = "integral",
        ebic_gamma: float = 0.0,
        adaptive_weights: bool = False,
        cross_weight_ratio: float = 5.0,
        enforce_effect_constraints: bool = True,
    ):
        self.max_order = max_order
        self.alpha = alpha
        self.mix = mix
        self.fix_mix = fix_mix
        self.nonneg_self = nonneg_self
        self.basis_decay = basis_decay
        self.max_interactions = max_interactions
        self.basis_type = basis_type
        self.ebic_gamma = ebic_gamma
        self.adaptive_weights = adaptive_weights
        self.cross_weight_ratio = float(cross_weight_ratio)
        self.enforce_effect_constraints = bool(enforce_effect_constraints)
        self.coef_: pd.DataFrame | None = None
        self.mse_: float | None = None
        self.bic_order_path_: pd.DataFrame | None = None
        self.bic_alpha_path_: pd.DataFrame | None = None
        self.power_function_params_: pd.DataFrame | None = None
        self.effect_constraint_directions_: pd.Series | None = None
        self.effect_constraint_diagnostics_: pd.DataFrame | None = None

    def _design(self, power_function_sample_df: pd.DataFrame) -> pd.DataFrame:
        basis = tiger_lop_basis_expansion(power_function_sample_df, self.max_order)
        if self.basis_decay > 0.0:
            n_feat = power_function_sample_df.shape[1]
            for r in range(self.max_order):
                scale = float(np.exp(-self.basis_decay * r))
                cols = [k * self.max_order + r for k in range(n_feat)]
                basis.iloc[:, cols] *= max(scale, 1e-10)
        intercept = pd.DataFrame(1.0, index=basis.index, columns=["intercept"])
        return pd.concat([intercept, basis], axis=1)

    def _group_size(self) -> int:
        """返回每个源变量在设计矩阵中的列数。"""
        return self.max_order

    def _build_self_cross_weights(
        self,
        p: int,
        groups: list[np.ndarray],
        n_targets: int,
        Xv: np.ndarray | None = None,
        Y: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """构造 per-target 的 ``(coef_weights, group_weights)``。

        - **结构权重**（``adaptive_weights=False``，默认）：self 组（``gi == j + 1``）
          系数权重 = 1，cross 组（其他非 intercept 组）系数权重 = ``cross_weight_ratio``;
          截距组权重 = 1（在 ``_asgl_col`` 内 ``gi == 0`` 不被惩罚，权重值仅作占位）。
        - **自适应权重**（``adaptive_weights=True``）：先用全设计矩阵 OLS 估初值
          ``W_ols``，再按 ``1 / (|·| + eps)`` 构造常规 ASGL 自适应权重，最后再乘以
          self/cross 区分比例（cross 额外乘 ``cross_weight_ratio``）。
        """
        ratio = self.cross_weight_ratio
        caws: list[np.ndarray] = []
        gaws: list[np.ndarray] = []
        eps = 1e-6

        if self.adaptive_weights and Xv is not None and Y is not None:
            W_ols, *_ = np.linalg.lstsq(Xv, Y, rcond=None)
        else:
            W_ols = None

        for j in range(n_targets):
            cw = np.ones(p, dtype=float)
            gw = np.ones(len(groups), dtype=float)
            for gi, g in enumerate(groups):
                if gi == 0:
                    continue
                base = 1.0 if gi == j + 1 else ratio
                if W_ols is not None:
                    elem_aw = 1.0 / (np.abs(W_ols[g, j]) + eps)
                    grp_aw = 1.0 / (float(np.linalg.norm(W_ols[g, j])) + eps)
                else:
                    elem_aw = np.ones(len(g), dtype=float)
                    grp_aw = 1.0
                cw[g] = base * elem_aw
                gw[gi] = base * grp_aw
            caws.append(cw)
            gaws.append(gw)
        return caws, gaws

    def _check_effect_constraints(
        self,
        Xv: np.ndarray,
        W: np.ndarray,
        groups: list[np.ndarray],
        target_names: list[str],
        target_indices: list[int] | None = None,
    ) -> tuple[bool, dict[str, str], list[dict[str, float | str | bool]]]:
        """检查自身效应相对总效应的逐点硬约束。"""
        if not self.enforce_effect_constraints:
            directions = {name: "not_enforced" for name in target_names}
            diagnostics = [
                {
                    "target": name,
                    "valid": True,
                    "direction": "not_enforced",
                    "reason": "not_enforced",
                    "min_delta": np.nan,
                    "max_delta": np.nan,
                    "mean_abs_delta": np.nan,
                    "self_dynamic_range": np.nan,
                    "total_range": np.nan,
                    "self_dynamic_frac": np.nan,
                }
                for name in target_names
            ]
            return True, directions, diagnostics

        directions: dict[str, str] = {}
        diagnostics: list[dict[str, float | str | bool]] = []
        all_valid = True
        gap_min = 1e-6
        mean_gap_min = 0.0

        if target_indices is None:
            target_indices = list(range(len(target_names)))

        for local_j, target_name in enumerate(target_names):
            j = target_indices[local_j]
            self_group = groups[j + 1]
            w_col = W[:, local_j] if W.shape[1] == len(target_names) else W[:, j]
            total_curve = Xv @ w_col
            self_dynamic = Xv[:, self_group] @ w_col[self_group]
            self_curve = w_col[0] + self_dynamic
            delta = self_curve - total_curve
            constraint_delta = delta[1:] if delta.shape[0] > 1 else delta

            min_delta = float(np.min(constraint_delta))
            max_delta = float(np.max(constraint_delta))
            mean_abs_delta = float(np.mean(np.abs(constraint_delta)))
            self_dynamic_range = _range(self_dynamic)
            total_range = _range(total_curve)
            if total_range > 1e-12:
                self_dynamic_frac = self_dynamic_range / total_range
            else:
                self_dynamic_frac = np.inf if self_dynamic_range > 0 else 0.0

            above_ok = min_delta >= gap_min
            below_ok = max_delta <= -gap_min
            if above_ok:
                direction = "self_above_total"
                direction_ok = True
            elif below_ok:
                direction = "self_below_total"
                direction_ok = True
            else:
                direction = "invalid"
                direction_ok = False

            reason = "ok"
            target_valid = True
            if not direction_ok:
                reason = "direction_violation"
                target_valid = False
            elif mean_abs_delta < mean_gap_min:
                reason = "mean_gap_too_small"
                target_valid = False

            if not target_valid:
                all_valid = False
            directions[target_name] = direction
            diagnostics.append(
                {
                    "target": target_name,
                    "valid": target_valid,
                    "direction": direction,
                    "reason": reason,
                    "min_delta": min_delta,
                    "max_delta": max_delta,
                    "mean_abs_delta": mean_abs_delta,
                    "self_dynamic_range": self_dynamic_range,
                    "total_range": total_range,
                    "self_dynamic_frac": float(self_dynamic_frac),
                }
            )

        return all_valid, directions, diagnostics

    def _repair_effect_constraints(
        self,
        Xv: np.ndarray,
        W: np.ndarray,
        groups: list[np.ndarray],
    ) -> np.ndarray:
        """用每个 cross 组的 offset 列把候选解推入方向可行域。"""
        if not self.enforce_effect_constraints or self._group_size() <= self.max_order:
            return W

        repaired = W.copy()
        gap_min = 1e-6
        repair_gap = gap_min * 1.01 + 1e-9
        for j in range(W.shape[1]):
            cross_groups = [
                groups[gi]
                for gi in range(1, len(groups))
                if gi != j + 1 and len(groups[gi]) > 0
            ]
            if not cross_groups:
                continue
            offset_col = int(cross_groups[0][0])
            offset_scale = float(Xv[0, offset_col])
            if abs(offset_scale) < 1e-12:
                continue

            cross_cols = np.concatenate(cross_groups)
            cross_sum = Xv[:, cross_cols] @ repaired[cross_cols, j]
            above_shift = -(float(np.max(cross_sum)) + repair_gap)
            below_shift = repair_gap - float(np.min(cross_sum))
            if abs(above_shift) <= abs(below_shift):
                shift = above_shift
            else:
                shift = below_shift
            repaired[offset_col, j] += shift / offset_scale
        return repaired

    def _select_lasso_cross_support(
        self,
        basis: np.ndarray,
        y_adj: np.ndarray,
        groups: list[np.ndarray],
        target_idx: int,
    ) -> list[int]:
        """先用 Lasso 选择 cross source 支撑集，再交给 CVXPY 精修。"""
        from sklearn.linear_model import Lasso

        candidate_group_ids = [
            gi for gi in range(1, len(groups)) if gi != target_idx + 1
        ]
        if not candidate_group_ids:
            return []

        cross_cols = np.concatenate([groups[gi] - 1 for gi in candidate_group_ids])
        X_cross = basis[:, cross_cols].astype(float, copy=True)
        col_norms = np.linalg.norm(X_cross, axis=0)
        keep_cols = col_norms > 1e-12
        if not np.any(keep_cols):
            return []

        X_scaled = X_cross[:, keep_cols] / col_norms[keep_cols]
        residual_centered = y_adj - float(np.mean(y_adj))
        n_samples = max(X_scaled.shape[0], 1)
        alpha_max = float(np.max(np.abs(X_scaled.T @ residual_centered))) / n_samples
        if alpha_max <= 1e-12:
            return []

        selected_local: np.ndarray | None = None
        last_coef: np.ndarray | None = None
        for alpha in np.geomspace(alpha_max, alpha_max * 1e-4, 30):
            lasso = Lasso(
                alpha=float(alpha),
                fit_intercept=False,
                max_iter=20000,
                tol=1e-6,
                selection="cyclic",
            )
            lasso.fit(X_scaled, residual_centered)
            last_coef = np.asarray(lasso.coef_, dtype=float)
            active = np.flatnonzero(np.abs(lasso.coef_) > 1e-8)
            if active.size > 0:
                selected_local = active
                break

        if selected_local is None:
            group_scores: list[tuple[int, float]] = []
            for gi in candidate_group_ids:
                cols = groups[gi] - 1
                score = float(np.linalg.norm(basis[:, cols].T @ residual_centered))
                group_scores.append((gi, score))
            group_scores.sort(key=lambda item: item[1], reverse=True)
            if not group_scores or group_scores[0][1] <= 1e-12:
                return []
            fallback_n = 1 if self.max_interactions <= 0 else int(self.max_interactions)
            return [gi for gi, _ in group_scores[:fallback_n]]

        kept_global_cols = cross_cols[keep_cols]
        selected_cols = set(kept_global_cols[selected_local].tolist())
        selected_groups = [
            gi
            for gi in candidate_group_ids
            if any(int(col - 1) in selected_cols for col in groups[gi])
        ]

        if self.max_interactions > 0 and len(selected_groups) > self.max_interactions:
            group_scores: list[tuple[int, float]] = []
            for gi in selected_groups:
                cols = groups[gi] - 1
                score = float(np.linalg.norm(basis[:, cols].T @ residual_centered))
                group_scores.append((gi, score))
            group_scores.sort(key=lambda item: item[1], reverse=True)
            selected_groups = [
                gi for gi, _ in group_scores[: int(self.max_interactions)]
            ]

        return selected_groups

    def _fit_one_order(
        self,
        power_function_sample_df: pd.DataFrame,
        quasi_dynamic_df: pd.DataFrame,
        intercept_values: np.ndarray | None,
    ) -> dict:
        """在当前 ``self.max_order`` 下做 TIGER-style 逐目标约束效应分解。"""
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise RuntimeError("TIGER-style constrained decomposition requires cvxpy") from exc

        X = self._design(power_function_sample_df)
        Y = align_response_to_design(quasi_dynamic_df, X.index).values.astype(float)
        Xv_raw = X.values.astype(float)
        basis = Xv_raw[:, 1:]
        n, p = Xv_raw.shape
        n_targets = Y.shape[1]
        target_names = [str(c) for c in quasi_dynamic_df.columns]
        group_size = self._group_size()
        n_vars = (p - 1) // group_size
        groups = _build_groups(n_vars, group_size)
        fix_intercept = intercept_values is not None
        curve_initial = align_response_to_design(
            power_function_sample_df.reindex(columns=quasi_dynamic_df.columns),
            X.index,
        ).values.astype(float)[0, :]
        y0_arr = (
            np.asarray(intercept_values, dtype=float)
            if fix_intercept
            else curve_initial
        )

        W = np.zeros((p, n_targets), dtype=float)
        W[0, :] = y0_arr
        alpha_rows: list[dict[str, float | int | bool]] = []
        ridge = 1e-6
        l1_scale = max(float(self.alpha), 1e-6) * 1e-4
        gap_min = 1e-6
        mean_gap_min = gap_min
        target_diagnostics: list[dict[str, float | str | bool]] = []
        target_directions: dict[str, str] = {}

        for j, target_name in enumerate(target_names):
            self_cols = groups[j + 1] - 1
            y_adj = Y[:, j] - y0_arr[j]
            cross_group_ids = self._select_lasso_cross_support(
                basis,
                y_adj,
                groups,
                j,
            )
            cross_cols = (
                np.concatenate([groups[gi] - 1 for gi in cross_group_ids])
                if cross_group_ids
                else np.array([], dtype=int)
            )
            active_cols = np.concatenate([self_cols, cross_cols])
            B_active = basis[:, active_cols]
            self_pos = np.arange(len(self_cols))
            cross_pos = np.arange(len(self_cols), len(active_cols))

            direction_options = ["self_above_total", "self_below_total"]
            best_obj = np.inf
            best_theta: np.ndarray | None = None
            best_direction = "invalid"
            best_diag: dict[str, float | str | bool] | None = None

            for direction in direction_options:
                theta = cp.Variable(len(active_cols))
                total_dynamic = B_active @ theta
                constraints = []
                if len(cross_pos) > 0 and self.enforce_effect_constraints:
                    cross_dynamic = B_active[:, cross_pos] @ theta[cross_pos]
                    constrained_cross = (
                        cross_dynamic[1:] if n > 1 else cross_dynamic
                    )
                    if direction == "self_above_total":
                        constraints.extend(
                            [
                                constrained_cross <= -gap_min,
                                cp.sum(-constrained_cross)
                                / max(n - 1, 1)
                                >= mean_gap_min,
                            ]
                        )
                    else:
                        constraints.extend(
                            [
                                constrained_cross >= gap_min,
                                cp.sum(constrained_cross)
                                / max(n - 1, 1)
                                >= mean_gap_min,
                            ]
                        )
                elif self.enforce_effect_constraints:
                    continue
                if self.nonneg_self:
                    self_dynamic_expr = B_active[:, self_pos] @ theta[self_pos]
                    constraints.append(y0_arr[j] + self_dynamic_expr >= 0.0)

                reg_terms = [ridge * cp.sum_squares(theta)]
                if len(cross_pos) > 0:
                    reg_terms.append(
                        l1_scale
                        * self.cross_weight_ratio
                        * cp.norm1(theta[cross_pos])
                    )
                objective = cp.Minimize(
                    cp.sum_squares(y_adj - total_dynamic) + sum(reg_terms)
                )
                problem = cp.Problem(objective, constraints)
                try:
                    problem.solve(solver=cp.CLARABEL, verbose=False)
                except cp.error.SolverError:
                    problem.solve(solver=cp.SCS, verbose=False)
                if theta.value is None or problem.status not in {
                    cp.OPTIMAL,
                    cp.OPTIMAL_INACCURATE,
                }:
                    continue

                theta_val = np.asarray(theta.value, dtype=float).reshape(-1)
                candidate_W = np.zeros((p, 1), dtype=float)
                candidate_W[0, 0] = y0_arr[j]
                candidate_W[1 + active_cols, 0] = theta_val
                valid, _, diag_rows = self._check_effect_constraints(
                    Xv_raw,
                    candidate_W,
                    groups,
                    [target_name],
                    [j],
                )
                if self.enforce_effect_constraints and not valid:
                    continue
                pred = Xv_raw @ candidate_W[:, 0]
                obj_val = float(np.sum((Y[:, j] - pred) ** 2))
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_theta = candidate_W[:, 0]
                    best_direction = direction
                    best_diag = diag_rows[0]

            if best_theta is None or best_diag is None:
                raise ValueError(
                    f"No feasible TIGER decomposition for target {target_name!r}. "
                    "Please lower thresholds or increase max_order/max_interactions."
                )
            W[:, j] = best_theta
            target_directions[target_name] = best_direction
            target_diagnostics.append(best_diag)

        rss_final = float(np.sum((Y - Xv_raw @ W) ** 2))
        df_final = int(np.sum(np.abs(W) > 1e-10))
        rss_floor = 1e-12 * n * n_targets
        bic_final = n * n_targets * np.log(
            max(rss_final, rss_floor) / max(n * n_targets, 1)
        ) + df_final * np.log(n)

        return {
            "X_columns": X.columns,
            "Y_columns": quasi_dynamic_df.columns,
            "W": W,
            "Xv_raw": Xv_raw,
            "Y": Y,
            "rss": rss_final,
            "df": df_final,
            "bic": bic_final,
            "alpha": self.alpha,
            "mix": self.mix,
            "alpha_rows": alpha_rows,
            "n": n,
            "n_targets": n_targets,
            "effect_constraint_directions": target_directions,
            "effect_constraint_diagnostics": target_diagnostics,
        }

    def _fit_asgl_bic(
        self,
        power_function_sample_df: pd.DataFrame,
        quasi_dynamic_df: pd.DataFrame,
        intercept_values: np.ndarray | None = None,
    ) -> "IDOPRegressor":
        """单阶段 self-dominant ASGL + 外层 BIC 选 ``max_order``。"""
        user_max_order = self.max_order
        upper = min(user_max_order, 20)
        lower = 1
        if upper < lower:
            upper = lower

        order_rows: list[dict[str, float | int | bool]] = []
        alpha_rows_all: list[dict[str, float | int | bool]] = []
        best_artifacts: dict | None = None
        best_order_bic = np.inf
        best_order = lower
        infeasible_messages: list[str] = []

        for order_c in range(lower, upper + 1):
            self.max_order = order_c
            try:
                res = self._fit_one_order(
                    power_function_sample_df, quasi_dynamic_df, intercept_values
                )
            except (np.linalg.LinAlgError, ValueError) as exc:
                infeasible_messages.append(f"max_order={order_c}: {exc}")
                order_rows.append(
                    {
                        "max_order": order_c,
                        "bic": np.nan,
                        "rss": np.nan,
                        "df": np.nan,
                        "n_obs": np.nan,
                        "n_targets": np.nan,
                    }
                )
                continue

            order_rows.append(
                {
                    "max_order": order_c,
                    "bic": float(res["bic"]),
                    "rss": float(res["rss"]),
                    "df": int(res["df"]),
                    "n_obs": int(res["n"]),
                    "n_targets": int(res["n_targets"]),
                }
            )
            for row in res["alpha_rows"]:
                row_with_order = dict(row)
                row_with_order["max_order"] = order_c
                alpha_rows_all.append(row_with_order)

            if res["bic"] < best_order_bic:
                best_order_bic = res["bic"]
                best_order = order_c
                best_artifacts = res

        if best_artifacts is None:
            detail = "; ".join(infeasible_messages[-3:])
            raise ValueError(
                "No feasible effect-decomposition solution satisfies the hard "
                "constraints. Please lower thresholds or increase max_order."
                + (f" Recent failures: {detail}" if detail else "")
            )

        self.max_order = best_order
        self.alpha = best_artifacts["alpha"]
        self.mix = best_artifacts["mix"]

        self.bic_order_path_ = pd.DataFrame(order_rows)
        if not self.bic_order_path_.empty:
            self.bic_order_path_["selected"] = (
                self.bic_order_path_["max_order"] == self.max_order
            )
        self.bic_alpha_path_ = pd.DataFrame(alpha_rows_all)
        if not self.bic_alpha_path_.empty:
            self.bic_alpha_path_["selected"] = (
                (self.bic_alpha_path_["max_order"] == self.max_order)
                & (self.bic_alpha_path_["alpha"] == self.alpha)
                & (self.bic_alpha_path_["mix"] == self.mix)
            )

        W = best_artifacts["W"]
        self.coef_ = pd.DataFrame(
            W,
            index=best_artifacts["X_columns"],
            columns=best_artifacts["Y_columns"],
        )
        self.mse_ = float(
            np.mean((best_artifacts["Y"] - best_artifacts["Xv_raw"] @ W) ** 2)
        )
        self.effect_constraint_directions_ = pd.Series(
            best_artifacts["effect_constraint_directions"],
            name="direction",
        )
        self.effect_constraint_diagnostics_ = pd.DataFrame(
            best_artifacts["effect_constraint_diagnostics"]
        )
        return self

    def fit(
        self,
        power_function_sample_df: pd.DataFrame,
        quasi_dynamic_df: pd.DataFrame,
        *,
        power_function_params: pd.DataFrame,
        intercept_values: np.ndarray | None = None,
    ) -> "IDOPRegressor":
        self.bic_order_path_ = None
        self.bic_alpha_path_ = None
        self.effect_constraint_directions_ = None
        self.effect_constraint_diagnostics_ = None
        self.power_function_params_ = power_function_params
        return self._fit_asgl_bic(
            power_function_sample_df, quasi_dynamic_df, intercept_values
        )

    def predict(self, power_function_sample_df: pd.DataFrame) -> pd.DataFrame:
        X = self._design(power_function_sample_df)
        return pd.DataFrame(
            X.values.astype(float) @ self.coef_.values,
            index=X.index,
            columns=self.coef_.columns,
        )

    def effect(self, power_function_sample_df: pd.DataFrame) -> list[pd.DataFrame]:
        """按每个源特征聚合积分基×系数，得到各目标上的源特征效应（不含截距）。"""
        if self.coef_ is None:
            raise RuntimeError("call fit before effect")
        X = self._design(power_function_sample_df)
        basis_int_df = X.drop(columns=["intercept"])
        n_feature = power_function_sample_df.shape[1]
        group_size = self._group_size()
        feature_names = list(power_function_sample_df.columns)
        effect_df_list: list[pd.DataFrame] = []
        for target in self.coef_.columns:
            coef_row = self.coef_.loc[basis_int_df.columns, target]
            weighted = basis_int_df.multiply(coef_row, axis=1)
            cols: list[pd.Series] = []
            for j in range(n_feature):
                cols.append(
                    weighted.iloc[:, j * group_size : (j + 1) * group_size].sum(
                        axis=1
                    )
                )
            collapsed = pd.concat(cols, axis=1)
            collapsed.columns = feature_names
            effect_df_list.append(collapsed)
        return effect_df_list

    def adjacency_matrix(
        self,
        power_function_sample_df: pd.DataFrame,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """计算邻接矩阵，可选按离散点均值或积分聚合基函数列。

        Parameters
        ----------
        power_function_sample_df : pd.DataFrame
            输入功率函数采样矩阵。
        aggregation : str, default "mean"
            邻接矩阵列聚合方式：
            - "mean"：对离散采样点直接取算术均值；
            - "integral"：按 index 作为自变量做梯形积分。
        """
        if self.coef_ is None:
            raise RuntimeError("call fit before adjacency_matrix")
        if aggregation not in ("mean", "integral"):
            raise ValueError("aggregation 必须为 'mean' 或 'integral'")
        X = self._design(power_function_sample_df)
        basis_int_df = X.drop(columns=["intercept"])
        m = power_function_sample_df.shape[1]
        group_size = self._group_size()
        targets = list(self.coef_.columns)
        names = list(power_function_sample_df.columns)
        x_axis = basis_int_df.index.to_numpy(dtype=float, copy=False)
        G = np.zeros((m, m), dtype=float)
        for r in range(group_size):
            theta_r = np.zeros((m, m), dtype=float)
            psi = np.zeros(m, dtype=float)
            for k in range(m):
                col_idx = k * group_size + r
                col_name = basis_int_df.columns[col_idx]
                theta_r[:, k] = self.coef_.loc[col_name, targets].values.astype(float)
                col_values = basis_int_df.iloc[:, col_idx].to_numpy(dtype=float, copy=False)
                if aggregation == "integral":
                    psi[k] = float(np.trapezoid(col_values, x=x_axis))
                else:
                    psi[k] = float(np.mean(col_values))
            G += theta_r * psi[np.newaxis, :]
        return pd.DataFrame(G.T, index=names, columns=targets)
