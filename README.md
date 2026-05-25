# FunClu-Semi

Semi-Analytical Accelerated Functional Clustering with Allometric Mean and SAD(1) Covariance.

## Overview

FunClu-Semi is a mixture-model-based functional clustering method that classifies high-dimensional longitudinal data by parameterizing both the mean function and covariance structure within each cluster. Key features:

- **Allometric power-law mean**: $\mu_{k,i}(t) = a_{k,i} \cdot t^{b_{k,i}}$ — 2 parameters per condition per cluster
- **SAD(1) covariance**: First-order structured antedependence model with tridiagonal inverse and closed-form determinant
- **Semi-analytical EM**: Closed-form WLS update for $a_{k,i}$, closed-form estimator for $\gamma_{k,i}^2$, $O(m)$ per-iteration complexity
- **Multivariate extension**: Block-diagonal covariance for multi-condition joint clustering
- **BIC-based model selection**: Automatic determination of the number of clusters

## Files

| File | Description |
|------|-------------|
| `funclu.py` | Core FunClu EM algorithm implementation (PyTorch) |
| `construction.py` | IDOPRegressor with Legendre basis expansion and ASGL |
| `plot.py` | Visualization utilities |
| `english_paper.tex` | English LaTeX manuscript |
| `chinese_paper.tex` | Chinese verification version |

## Dependencies

- Python 3.10+
- PyTorch
- scikit-learn
- NumPy, pandas, scipy

## Quick Start

```python
from funclu import FunClu, scan_bic

# Prepare data: list of DataFrames (one per condition)
data = [df_condition1, df_condition2, ...]  # shape: (n_features, n_timepoints)

# Fit model
model = FunClu(n_components=5)
model.fit(data)

# Get cluster labels
labels = model.predict(data)

# BIC-based K selection
bic_results = scan_bic(data, k_min=2, k_max=10)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2026funclu,
  title={A Semi-Analytical Accelerated Functional Clustering Method with Allometric Mean and SAD(1) Covariance},
  author={...},
  year={2026}
}
```

## License

MIT
