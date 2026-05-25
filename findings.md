# 研究发现与文献要点

## 1. FunClu 核心框架 (Pan et al. 2026)

### 模型结构
- **混合模型**: K 个高斯分量，每个分量有参数化均值和协方差
- **均值建模**: 基于异速生长律 (allometric scaling law)
  - 栖息地指数 (Habitat Index): E_i = Σ_j y_{ij}
  - 幂函数均值: μ_k(E_i) = a_k · E_i^{b_k}
- **协方差建模**: SAD(1) 结构化协方差
  - Σ = A D A', 其中 A 为下三角矩阵，D 为创新方差对角阵
  - 参数: φ (自回归系数), ν² (创新方差)
- **EM 算法**: 混合 EM + Adam 数值优化
  - E步: 计算后验概率 ω_{i,k}
  - M步: α_k 闭式解 + softmax 平滑; (a_k, b_k, φ, ν) Adam 梯度优化
- **Multivariate 扩展**: 多组织/条件下共享协方差结构
- **模型选择**: BIC 确定最优聚类数 K

### 关键创新
1. 异速生长均值结构使 FunClu 适用于非时序静态高维数据
2. Softmax 平滑防止空簇崩溃
3. 混合优化策略平衡解析精度与数值灵活性

## 2. Functional Mapping 起源 (Ma et al. 2002)

- 首次将生物生长定律（Logistic 曲线）融入 QTL 定位框架
- 提出"功能映射"(Functional Mapping) 概念
- 通过估计少量生物意义参数替代大量均值向量元素
- 使用 EM 算法估计参数，AR(1) 建模残差协方差

## 3. LOP 函数聚类 (Wang et al. 2014)

- Legendre 正交多项式 (LOP) 非参数拟合动态表达曲线
- 混合模型框架 + EM + Simplex 算法
- SAD 模型建模纵向协方差
- 应用于蛋白质组动态数据

## 4. SAD 协方差模型 (Núñez-Antón 1997)

### AD(1) 模型性质
- 一阶前依赖模型: ρ_{ij} = Π_{l=i}^{j-1} ρ_{l,l+1}
- 协方差矩阵完全由 (2p-1) 个参数确定
- 行列式和逆矩阵有闭式解（三对角矩阵）
- Box-Cox 时间尺度变换产生非平稳结构
- SAD(1) 特例: ρ_i = φ^{f(t_{i+1},λ) - f(t_i,λ)}, σ_i² = g(t_i; ψ)

## 5. Python 加速算法实现 (construction.py)

### IDOPRegressor 核心算法
- **基函数扩展**: Legendre 多项式积分基 (TIGER/LOP 风格)
- **解析积分**: 对幂函数 y_k(τ) = a_k · τ^{b_k} 的 Legendre 基积分解析求值
- **稀疏回归**: Adaptive Sparse Group Lasso (ASGL) 近端梯度下降
  - self/cross 区分惩罚权重 (cross_weight_ratio)
  - nonneg_self 投影约束
- **BIC 选阶**: 外层遍历 max_order, 内层 CVXPY 约束优化做效应分解
- **效应约束**: self 效应方向性硬约束 (self_above_total / self_below_total)

### 关键加速技术
1. 解析积分替代梯形数值积分（消除离散误差）
2. 列 L2 归一化加速收敛
3. 预计算伪逆复用投影步骤
4. 分组稀疏结构降低计算复杂度

## 6. 模拟结果要点 (Pan et al. 2026)

- FunClu > GMM > KMeans（所有噪声水平下）
- FunClu_Smoothing 防止空簇，稳定性更优
- 高相关性 (φ=0.9) 下所有方法表现更好
- 低噪声 (ν≤0.4) 时 FunClu 优势最显著
- CA 最高达 0.91（φ=0.9, ν=0.2, FunClu_Smoothing）
