"""FunClu-Semi simulation study — v3: computational, interpretability, model selection.

Demonstrates advantages over K-means/GMM:
1. Computational: O(m) vs O(m^2) as m grows
2. Interpretability: recovers allometric exponent b with meaningful accuracy
3. Model selection: BIC correctly selects number of clusters
4. Small-sample: structured model outperforms GMM when N << m^2
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

import sys, torch
sys.path.insert(0, "/Users/yu/research/FunClu")
from funclu import FunClu

warnings.filterwarnings("ignore")


def generate_sad1_single(
    n_features: int, m: int,
    a_true: np.ndarray, b_true: np.ndarray,
    phi: float, nu: float,
    labels: np.ndarray, t: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    n_clusters = len(a_true)
    mu = np.zeros((n_features, m))
    for k in range(n_clusters):
        mask = labels == k
        mu[mask] = a_true[k] * t ** b_true[k]
    noise = np.zeros((n_features, m))
    for j in range(n_features):
        eps = rng.normal(0, 1, m)
        y = np.zeros(m)
        y[0] = nu * eps[0]
        for i in range(1, m):
            y[i] = phi * y[i - 1] + nu * eps[i]
        noise[j] = y
    return mu + noise


# ═══════════════════════════════════════════════════════════════
# 1. Timing benchmark: O(m) vs O(m^2)
# ═══════════════════════════════════════════════════════════════
def run_timing_benchmark() -> pd.DataFrame:
    a_true = np.array([0.5, 1.0, 2.0])
    b_true = np.array([0.3, 1.2, -0.5])
    phi, nu = 0.7, 0.4
    n_features = 200
    n_reps = 15

    rows = []
    for m in [5, 10, 20, 50, 100, 200]:
        t = np.linspace(1, 10, m)
        for rep in range(n_reps):
            seed = rep + 9999
            rng = np.random.RandomState(seed)
            labels = rng.choice(3, size=n_features, p=[0.3, 0.4, 0.3])
            data = generate_sad1_single(
                n_features, m, a_true, b_true, phi, nu, labels, t, rng
            )
            df = pd.DataFrame(data.T, index=t)
            row = {"m": m}

            t0 = time.time()
            try:
                m_fclu = FunClu(n_components=3, max_iter=30, tol=1e-4, random_state=seed)
                m_fclu.fit([df])
                row["funclu_time"] = time.time() - t0
            except:
                row["funclu_time"] = float("nan")

            t0 = time.time()
            try:
                KMeans(n_clusters=3, n_init=5, random_state=seed).fit_predict(data)
                row["kmeans_time"] = time.time() - t0
            except:
                row["kmeans_time"] = float("nan")

            t0 = time.time()
            try:
                GaussianMixture(n_components=3, covariance_type="full",
                                max_iter=50, random_state=seed,
                                n_init=2, reg_covar=1e-4).fit(data)
                row["gmm_time"] = time.time() - t0
            except:
                row["gmm_time"] = float("nan")

            rows.append(row)
        print(f"  timing: m={m:3d} done", flush=True)

    df = pd.DataFrame(rows)
    return df.groupby("m")[["funclu_time", "kmeans_time", "gmm_time"]].mean().reset_index()


# ═══════════════════════════════════════════════════════════════
# 2. Parameter recovery
# ═══════════════════════════════════════════════════════════════
def run_parameter_recovery(n_reps: int = 50) -> Dict:
    a_true = np.array([0.5, 1.0, 2.0])
    b_true = np.array([0.3, 1.2, -0.5])
    phi_true = 0.7
    nu_true = 0.3
    n_features = 600
    m = 20
    t = np.linspace(1, 10, m)

    a_rmse_list, b_rmse_list = [], []
    phi_rmse_list, gamma_rmse_list = [], []
    ari_list = []

    for rep in range(n_reps):
        seed = rep + 2000
        rng = np.random.RandomState(seed)
        labels = rng.choice(3, size=n_features, p=[0.3, 0.4, 0.3])
        data = generate_sad1_single(
            n_features, m, a_true, b_true, phi_true, nu_true, labels, t, rng
        )
        df = pd.DataFrame(data.T, index=t)

        try:
            model = FunClu(n_components=3, max_iter=50, tol=1e-4, random_state=seed)
            model.fit([df])
            pred = model.labels.cpu().numpy().ravel()
            ari = adjusted_rand_score(labels, pred)
            ari_list.append(ari)

            mu_est = model.params_mu.cpu().numpy()
            cov_est = model.params_cov.cpu().numpy()

            contingency = np.zeros((3, 3))
            for ek in range(3):
                for tk in range(3):
                    contingency[ek, tk] = np.sum((pred == ek) & (labels == tk))
            row_ind, col_ind = linear_sum_assignment(-contingency)
            mapping = dict(zip(row_ind, col_ind))

            a_est_list, b_est_list = [], []
            phi_est_list, gamma_est_list = [], []
            for k in range(3):
                tk = mapping.get(k, k)
                a_est_list.append(mu_est[k, 0, 0])
                b_est_list.append(mu_est[k, 0, 1])
                phi_est_list.append(cov_est[k, 0, 0])
                gamma_est_list.append(cov_est[k, 0, 1])

            a_rmse = np.sqrt(np.mean((np.array(a_est_list) - a_true) ** 2))
            b_rmse = np.sqrt(np.mean((np.array(b_est_list) - b_true) ** 2))
            phi_rmse = np.sqrt(np.mean((np.array(phi_est_list) - phi_true) ** 2))
            gamma_rmse = np.sqrt(np.mean((np.array(gamma_est_list) - nu_true) ** 2))

            a_rmse_list.append(a_rmse)
            b_rmse_list.append(b_rmse)
            phi_rmse_list.append(phi_rmse)
            gamma_rmse_list.append(gamma_rmse)

        except Exception as e:
            print(f"  Param recovery rep {rep}: {e}")

    return {
        "a_rmse_mean": np.mean(a_rmse_list), "a_rmse_std": np.std(a_rmse_list),
        "b_rmse_mean": np.mean(b_rmse_list), "b_rmse_std": np.std(b_rmse_list),
        "phi_rmse_mean": np.mean(phi_rmse_list), "phi_rmse_std": np.std(phi_rmse_list),
        "gamma_rmse_mean": np.mean(gamma_rmse_list), "gamma_rmse_std": np.std(gamma_rmse_list),
        "ari_mean": np.mean(ari_list), "ari_std": np.std(ari_list),
    }


# ═══════════════════════════════════════════════════════════════
# 3. BIC model selection
# ═══════════════════════════════════════════════════════════════
def run_bic_selection(n_reps: int = 30) -> Dict:
    a_true = np.array([0.5, 1.0, 2.0])
    b_true = np.array([0.3, 1.2, -0.5])
    n_features = 300
    m = 20
    t = np.linspace(1, 10, m)

    correct = 0
    k_choices = []
    for rep in range(n_reps):
        rng = np.random.RandomState(rep + 5000)
        labels = rng.choice(3, size=n_features, p=[0.3, 0.4, 0.3])
        data = generate_sad1_single(
            n_features, m, a_true, b_true, 0.7, 0.3, labels, t, rng
        )
        df = pd.DataFrame(data.T, index=t)

        best_bic = float("inf")
        best_k = 0
        for k_cand in [2, 3, 4, 5, 6]:
            try:
                model = FunClu(n_components=k_cand, max_iter=50, tol=1e-4, random_state=rep)
                model.fit([df])
                if model.bic is not None and model.bic < best_bic:
                    best_bic = model.bic
                    best_k = k_cand
            except:
                pass

        if best_k == 3:
            correct += 1
        k_choices.append(best_k)

    return {"bic_accuracy": correct / n_reps, "k_choices": k_choices}


# ═══════════════════════════════════════════════════════════════
# 4. Small-sample efficiency
# ═══════════════════════════════════════════════════════════════
def run_small_sample(n_reps: int = 50) -> Dict:
    a_true = np.array([0.5, 1.0, 2.0])
    b_true = np.array([0.3, 1.2, -0.5])
    phi_true = 0.7
    nu_true = 0.4
    n_features = 30
    m = 15
    t = np.linspace(1, 10, m)

    funclu_ari = []
    gmm_ari = []
    kmeans_ari = []

    for rep in range(n_reps):
        seed = rep + 8000
        rng = np.random.RandomState(seed)
        labels = rng.choice(3, size=n_features, p=[0.3, 0.4, 0.3])
        data = generate_sad1_single(
            n_features, m, a_true, b_true, phi_true, nu_true, labels, t, rng
        )
        df = pd.DataFrame(data.T, index=t)

        try:
            model = FunClu(n_components=3, max_iter=50, tol=1e-4, random_state=seed)
            model.fit([df])
            pred = model.labels.cpu().numpy().ravel()
            funclu_ari.append(adjusted_rand_score(labels, pred))
        except:
            pass

        try:
            gmm = GaussianMixture(n_components=3, covariance_type="full",
                                  max_iter=100, random_state=seed,
                                  n_init=5, reg_covar=1e-4)
            pred = gmm.fit_predict(data)
            gmm_ari.append(adjusted_rand_score(labels, pred))
        except:
            pass

        try:
            km = KMeans(n_clusters=3, n_init=10, random_state=seed)
            pred = km.fit_predict(data)
            kmeans_ari.append(adjusted_rand_score(labels, pred))
        except:
            pass

    return {
        "funclu_ari_mean": np.mean(funclu_ari), "funclu_ari_std": np.std(funclu_ari),
        "gmm_ari_mean": np.mean(gmm_ari), "gmm_ari_std": np.std(gmm_ari),
        "kmeans_ari_mean": np.mean(kmeans_ari), "kmeans_ari_std": np.std(kmeans_ari),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    sep = "=" * 70
    print(sep, flush=True)

    # 1. Timing benchmark
    print("\n[1/4] Timing benchmark (varying m)", flush=True)
    print("-" * 70, flush=True)
    timing_df = run_timing_benchmark()
    print(timing_df.to_string(float_format=lambda x: f"{x:.5f}"), flush=True)

    # 2. Parameter recovery
    print("\n[2/4] Parameter recovery", flush=True)
    print("-" * 70, flush=True)
    pr = run_parameter_recovery(n_reps=50)
    print(f"  ARI:          {pr['ari_mean']:.3f} (+-{pr['ari_std']:.3f})", flush=True)
    print(f"  a RMSE:       {pr['a_rmse_mean']:.4f} (+-{pr['a_rmse_std']:.4f})", flush=True)
    print(f"  b RMSE:       {pr['b_rmse_mean']:.4f} (+-{pr['b_rmse_std']:.4f})", flush=True)
    print(f"  phi RMSE:     {pr['phi_rmse_mean']:.4f} (+-{pr['phi_rmse_std']:.4f})", flush=True)
    print(f"  gamma RMSE:   {pr['gamma_rmse_mean']:.4f} (+-{pr['gamma_rmse_std']:.4f})", flush=True)

    # 3. BIC model selection
    print("\n[3/4] BIC model selection (K=3)", flush=True)
    print("-" * 70, flush=True)
    bic = run_bic_selection(n_reps=30)
    k_counts = pd.Series(bic["k_choices"]).value_counts().sort_index()
    print(f"  BIC accuracy: {bic['bic_accuracy']*100:.0f}%", flush=True)
    print(f"  K choices: {dict(k_counts)}", flush=True)

    # 4. Small-sample efficiency
    print("\n[4/4] Small-sample efficiency (N=30, m=15)", flush=True)
    print("-" * 70, flush=True)
    ss = run_small_sample(n_reps=50)
    print(f"  FunClu-Semi: {ss['funclu_ari_mean']:.3f} (+-{ss['funclu_ari_std']:.3f})", flush=True)
    print(f"  GMM:         {ss['gmm_ari_mean']:.3f} (+-{ss['gmm_ari_std']:.3f})", flush=True)
    print(f"  K-means:     {ss['kmeans_ari_mean']:.3f} (+-{ss['kmeans_ari_std']:.3f})", flush=True)

    # LaTeX tables
    print("\n" + sep, flush=True)
    print("\n--- TIMING TABLE ---", flush=True)
    print("\\begin{table}[htbp]", flush=True)
    print("    \\centering", flush=True)
    print("    \\caption{Mean runtime (seconds) vs number of time points $m$}", flush=True)
    print("    \\label{tab:timing}", flush=True)
    print("    \\begin{tabular}{cccc}", flush=True)
    print("        \\toprule", flush=True)
    print("        $m$ & FunClu-Semi & K-means & GMM \\\\", flush=True)
    print("        \\midrule", flush=True)
    for _, row in timing_df.iterrows():
        print(f"        {int(row['m'])} & {row['funclu_time']:.5f} & {row['kmeans_time']:.5f} & {row['gmm_time']:.5f} \\\\", flush=True)
    print("        \\bottomrule", flush=True)
    print("    \\end{tabular}", flush=True)
    print("\\end{table}", flush=True)

    print("\n--- PARAMETER RECOVERY TABLE ---", flush=True)
    print("\\begin{table}[htbp]", flush=True)
    print("    \\centering", flush=True)
    print("    \\caption{Parameter recovery RMSE for FunClu-Semi (N=600, m=20, $\\phi=0.7$, $\\nu=0.3$, 50 reps)}", flush=True)
    print("    \\label{tab:param_recovery}", flush=True)
    print("    \\begin{tabular}{lccc}", flush=True)
    print("        \\toprule", flush=True)
    print("        Parameter & True value & RMSE & Std \\\\", flush=True)
    print("        \\midrule", flush=True)
    print(f"        $a$ & (0.5, 1.0, 2.0) & {pr['a_rmse_mean']:.4f} & ({pr['a_rmse_std']:.4f}) \\\\", flush=True)
    print(f"        $b$ & (0.3, 1.2, -0.5) & {pr['b_rmse_mean']:.4f} & ({pr['b_rmse_std']:.4f}) \\\\", flush=True)
    print(f"        $\\phi$ & 0.7 & {pr['phi_rmse_mean']:.4f} & ({pr['phi_rmse_std']:.4f}) \\\\", flush=True)
    print(f"        $\\gamma$ & 0.3 & {pr['gamma_rmse_mean']:.4f} & ({pr['gamma_rmse_std']:.4f}) \\\\", flush=True)
    print("        \\bottomrule", flush=True)
    print("    \\end{tabular}", flush=True)
    print("\\end{table}", flush=True)

    print("\n--- SMALL SAMPLE TABLE ---", flush=True)
    print("\\begin{table}[htbp]", flush=True)
    print("    \\centering", flush=True)
    print("    \\caption{Clustering ARI under small-sample setting (N=30, m=15, 50 reps)}", flush=True)
    print("    \\label{tab:small_sample}", flush=True)
    print("    \\begin{tabular}{lcc}", flush=True)
    print("        \\toprule", flush=True)
    print("        Method & Mean ARI & Std \\\\", flush=True)
    print("        \\midrule", flush=True)
    print(f"        FunClu-Semi & {ss['funclu_ari_mean']:.3f} & {ss['funclu_ari_std']:.3f} \\\\", flush=True)
    print(f"        GMM (full cov) & {ss['gmm_ari_mean']:.3f} & {ss['gmm_ari_std']:.3f} \\\\", flush=True)
    print(f"        K-means & {ss['kmeans_ari_mean']:.3f} & {ss['kmeans_ari_std']:.3f} \\\\", flush=True)
    print("        \\bottomrule", flush=True)
    print("    \\end{tabular}", flush=True)
    print("\\end{table}", flush=True)


if __name__ == "__main__":
    main()
