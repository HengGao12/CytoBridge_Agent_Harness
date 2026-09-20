# DeepRUOT Perturbation & Trajectory Generation 实现文档

本文档详细说明了 CytoBridge 中基因扰动分析和轨迹生成模块的实现逻辑。

---

## 目录

1. [概述](#概述)
2. [基因扰动分析](#基因扰动分析)
3. [SDE 轨迹模拟](#sde-轨迹模拟)
4. [细胞类型分类器](#细胞类型分类器)
5. [轨迹数据集生成与导出](#轨迹数据集生成与导出)
6. [空间转换: Latent ↔ Gene](#空间转换)
7. [完整使用示例](#完整使用示例)

---

## 概述

该模块位于 `CytoBridge/tl/perturbation.py`，包含以下核心功能：

| 类别 | 函数 | 功能 |
|------|------|------|
| **扰动** | `perturb_gene_expression()` | 对指定基因施加 z-score 扰动 |
| **模拟** | `simulate_perturbation_sde()` | 使用 SDE 模拟细胞轨迹 |
| **分类** | `train_cell_classifier()` | 训练 MLP 细胞类型分类器 |
| **分类** | `classify_final_states()` | 分类最终细胞状态 |
| **导出** | `generate_trajectory_dataset()` | 生成并导出轨迹数据集 |

---

## 基因扰动分析

### 原理

基因扰动通过以下步骤实现：

```
原始表达矩阵 X (n_cells, n_genes)
    ↓ Z-score 标准化
[X - μ] / σ  →  X_scaled
    ↓ 对目标基因设置特定 z 值
X_scaled[:, gene_idx] = z_score
    ↓ PCA 投影到潜在空间
X_perturbed @ PCs  →  X_latent_perturbed
```

### 代码实现

```python
# CytoBridge/tl/perturbation.py: perturb_gene_expression()

def perturb_gene_expression(
    adata: AnnData,
    genes: List[str],           # 目标基因列表
    z_score: float,             # 扰动强度 (-2, -1, 0, 1, 2 等)
    pca_model: Optional[Any] = None,
    use_stored_pca: bool = True,
) -> np.ndarray:
    
    # 1. 验证基因存在性
    gene_names = list(adata.var_names)
    valid_genes = [g for g in genes if g in gene_names]
    
    # 2. 获取并标准化表达矩阵
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. 应用扰动: 将目标基因设为特定 z-score
    X_perturbed = X_scaled.copy()
    gene_indices = [gene_names.index(g) for g in valid_genes]
    for idx in gene_indices:
        X_perturbed[:, idx] = z_score  # 关键: 设置为固定 z 值
    
    # 4. 投影到潜在空间
    if use_stored_pca and 'PCs' in adata.varm:
        pcs = adata.varm['PCs']
        x_perturb = X_perturbed @ pcs[:, :latent_dim]
    
    return x_perturb.astype(np.float32)
```

### Z-Score 含义

| z-score | 生物学意义 |
|---------|-----------|
| z > 0 | 基因上调 (高于平均表达) |
| z = 0 | 正常表达 (对照) |
| z < 0 | 基因下调 (低于平均表达) |
| z = 2 | 高于平均值 2 个标准差 |
| z = -2 | 低于平均值 2 个标准差 |

---

## SDE 轨迹模拟

### 数学原理

使用随机微分方程 (SDE) 模拟细胞动态：

```
dX = f(t, X)dt + σ·dW

其中:
- f(t, X) = velocity(t, X) + score_gradient(t, X)  # 漂移项
- σ = diffusion coefficient                          # 扩散系数
- dW = Wiener process increment                      # 随机噪声
```

### 代码实现

```python
# CytoBridge/tl/perturbation.py: _CytoBridgeSDE

class _CytoBridgeSDE(nn.Module):
    """封装 CytoBridge DynamicalModel 的 SDE"""
    noise_type = "diagonal"
    sde_type = "ito"
    
    def __init__(self, model: nn.Module, sigma: float = 0.1):
        super().__init__()
        self.model = model
        self.sigma = sigma
    
    def f(self, t, y):
        """漂移函数"""
        z, lnw = y
        
        # 从模型获取速度场和评分
        outputs = self.model(t.expand(n, 1), z, lnw, except_interaction=True)
        drift = outputs.get('velocity', torch.zeros_like(z))
        
        # 添加 score gradient (如果有)
        if 'score_gradient' in outputs:
            drift = drift + outputs['score_gradient']
        
        # 生长率更新 log-weight
        dlnw = outputs.get('growth', torch.zeros_like(lnw))
        
        return drift, dlnw
    
    def g(self, t, y):
        """扩散函数"""
        z, lnw = y
        return torch.ones_like(z) * self.sigma, torch.zeros_like(lnw)


# Euler-Maruyama 积分
def euler_sde_integrate(sde, y0, ts, dt=0.1):
    z, lnw = y0
    z_traj = [z.clone()]
    
    for i in range(1, len(ts)):
        step = dt
        
        # 漂移项
        drift_z, drift_lnw = sde.f(current_t, (z, lnw))
        
        # 扩散项
        diff_z, _ = sde.g(current_t, (z, lnw))
        
        # Euler-Maruyama 更新
        noise = torch.randn_like(z)
        z = z + drift_z * step + diff_z * np.sqrt(step) * noise
        lnw = lnw + drift_lnw * step
        
        z_traj.append(z.clone())
    
    return torch.stack(z_traj)
```

### SDE vs ODE

| 方法 | 特点 | 适用场景 |
|------|------|---------|
| **ODE** | 确定性，无噪声 | 平滑轨迹，平均行为 |
| **SDE** | 随机性，有噪声 | 细胞命运随机性分析 |

---

## 细胞类型分类器

### MLP 架构

```python
class MLPClassifier(nn.Module):
    """2层隐藏层 MLP"""
    
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),      # 输入层
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)    # 输出层
        )
    
    def forward(self, x):
        return self.net(x)
```

### 训练流程

```python
def train_cell_classifier(adata, label_key, ...):
    
    # 1. 准备数据
    X = np.array(adata.obsm['X_latent'])  # 潜在表示
    y = adata.obs[label_key].values        # 细胞类型标签
    
    # 2. 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. 创建模型
    model = MLPClassifier(in_dim=X.shape[1], n_classes=len(le.classes_))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 4. 训练循环
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
    
    # 5. 保存模型
    torch.save({
        'model': model.state_dict(),
        'label_encoder': le,
        'classes': list(le.classes_),
        'in_dim': in_dim,
        'hidden_dim': hidden_dim,
    }, save_path)
    
    return {'model': model, 'label_encoder': le, ...}
```

### 使用分类器

```python
from CytoBridge.tl.perturbation import load_mlp_classifier

# 加载
model, label_encoder = load_mlp_classifier('path/to/classifier.pt')

# 预测
with torch.no_grad():
    x = torch.tensor(latent_positions)
    logits = model(x)
    pred = logits.argmax(dim=1)
    labels = label_encoder.inverse_transform(pred.numpy())
```

---

## 轨迹数据集生成与导出

### 数据格式

```
trajectories/
├── trajectory_dataset.npz     # NumPy 压缩文件
│   ├── trajectories_latent    # shape: (n_steps+1, n_cells, latent_dim)
│   ├── trajectories_gene      # shape: (n_steps+1, n_cells, n_genes)
│   ├── time_points            # shape: (n_steps+1,)
│   └── init_indices           # shape: (n_cells,)
├── gene_names.txt             # 每行一个基因名
└── trajectory_metadata.json   # 配置信息
```

### 代码实现

```python
def generate_trajectory_dataset(
    adata: AnnData,
    model: nn.Module,
    n_cells: int = 100,
    n_steps: int = 50,
    method: str = 'sde',
    sigma: float = 0.1,
    output_space: str = 'both',
    save_dir: Optional[str] = None,
    ...
) -> Dict[str, Any]:
    
    # 1. 准备数据
    x_latent = np.array(adata.obsm['X_latent']).astype(np.float32)
    
    # 2. 创建时间网格
    ts = torch.linspace(init_time, end_time, n_steps + 1)
    
    # 3. 采样初始细胞
    init_idx = np.random.choice(eligible_indices, size=n_cells, replace=False)
    x0 = torch.tensor(x_latent[init_idx])
    
    # 4. 运行模拟
    if method == 'sde':
        sde = _CytoBridgeSDE(model, sigma=sigma)
        traj_latent, _ = euler_sde_integrate(sde, (x0, lnw0), ts)
    else:
        # ODE: 简化 Euler 方法
        for i in range(n_steps):
            velocity = model(t, z, lnw)['velocity']
            z = z + velocity * dt
    
    # 5. 投影到基因空间
    if output_space in ['gene', 'both']:
        if 'PCs' in adata.varm:
            pcs = adata.varm['PCs'][:, :latent_dim]
            
            # 展平 → 投影 → 重塑
            n_t, n_c, n_d = traj_latent.shape
            traj_flat = traj_latent.reshape(-1, n_d)
            traj_gene_flat = traj_flat @ pcs.T  # 关键: 使用 PCA 逆变换
            traj_gene = traj_gene_flat.reshape(n_t, n_c, -1)
    
    # 6. 保存数据
    if save_dir:
        # 保存 NPZ
        np.savez(save_path, 
                 trajectories_latent=traj_latent,
                 trajectories_gene=traj_gene,
                 time_points=time_points,
                 init_indices=init_idx)
        
        # 保存基因名
        with open(gene_names_path, 'w') as f:
            for g in adata.var_names:
                f.write(f"{g}\n")
        
        # 保存元数据
        json.dump(metadata, open(metadata_path, 'w'))
    
    return results
```

---

## 空间转换

### Latent → Gene (PCA 逆变换)

```python
# 原理: X_gene = X_latent @ PCs.T
# 其中 PCs 是 PCA loadings, shape: (n_genes, n_components)

def project_latent_to_gene(latent_vectors, pcs):
    """
    latent_vectors: (n_samples, latent_dim)
    pcs: (n_genes, latent_dim)
    returns: (n_samples, n_genes)
    """
    return latent_vectors @ pcs.T
```

### Gene → Latent (PCA 正变换)

```python
# 原理: X_latent = X_gene_scaled @ PCs
# 其中 X_gene_scaled 是 z-score 标准化后的基因表达

def project_gene_to_latent(gene_expression, pcs, scaler=None):
    """
    gene_expression: (n_samples, n_genes)
    pcs: (n_genes, latent_dim)  
    returns: (n_samples, latent_dim)
    """
    if scaler:
        gene_scaled = scaler.transform(gene_expression)
    else:
        gene_scaled = (gene_expression - gene_expression.mean(axis=0)) / gene_expression.std(axis=0)
    
    return gene_scaled @ pcs
```

### 分类器兼容性

分类器训练和预测都在 **Latent 空间** 进行：

```python
# 训练时
X_train = adata.obsm['X_latent']  # Latent 空间

# 预测轨迹最终状态
final_latent = trajectories_latent[-1]  # (n_cells, latent_dim)
labels = classifier.predict(final_latent)

# 如果需要从基因空间轨迹预测，需要先投影
final_gene = trajectories_gene[-1]  # (n_cells, n_genes)
final_latent = project_gene_to_latent(final_gene, pcs)
labels = classifier.predict(final_latent)
```

---

## 完整使用示例

### 示例 1: 基因扰动分析

```python
import scanpy as sc
from CytoBridge.utils import load_model_from_adata
from CytoBridge.tl.perturbation import (
    perturb_gene_expression,
    simulate_perturbation_sde,
    classify_final_states,
)

# 1. 加载数据和模型
adata = sc.read_h5ad('trained_model.h5ad')
model = load_model_from_adata(adata)

# 2. 对 Klf4 基因上调 (z=2)
x_perturbed = perturb_gene_expression(adata, genes=['Klf4'], z_score=2.0)

# 3. 模拟轨迹
result = simulate_perturbation_sde(
    model=model,
    adata=adata,
    x_perturbed=x_perturbed,
    n_simulations=20,
    n_init_cells=100,
    n_steps=40,
    sigma=0.1,
)

# 4. 分类最终状态
final_positions = result['perturbed_trajectories'][:, -1, :, :]
labels, proportions = classify_final_states(
    final_positions, adata,
    label_key='cell_type',
    method='knn'
)

print("细胞类型比例:")
for cell_type, prop in proportions.items():
    print(f"  {cell_type}: {prop*100:.1f}%")
```

### 示例 2: 生成轨迹数据集

```python
from CytoBridge.tl.perturbation import (
    generate_trajectory_dataset,
    load_trajectory_dataset,
    train_cell_classifier,
    load_mlp_classifier,
)

# 1. 训练分类器
result = train_cell_classifier(
    adata=adata,
    label_key='cell_type',
    epochs=100,
    save_path='classifier.pt'
)
print(f"验证准确率: {result['val_accuracy']*100:.2f}%")

# 2. 生成轨迹
traj_result = generate_trajectory_dataset(
    adata=adata,
    model=model,
    n_cells=200,
    n_steps=50,
    method='sde',
    output_space='both',
    save_dir='trajectories/'
)

# 3. 加载轨迹
data = load_trajectory_dataset('trajectories/')
traj_gene = data['trajectories_gene']  # (51, 200, n_genes)
traj_latent = data['trajectories_latent']
gene_names = data['gene_names']

# 4. 分析特定基因轨迹
idx = gene_names.index('Klf4')
klf4_traj = traj_gene[:, :, idx]  # (51, 200)

import matplotlib.pyplot as plt
plt.plot(data['time_points'], klf4_traj.mean(axis=1))
plt.xlabel('Time')
plt.ylabel('Klf4 Expression')
plt.title('Average Klf4 Expression Over Time')
plt.savefig('klf4_trajectory.png')

# 5. 注释最终细胞状态
model, le = load_mlp_classifier('classifier.pt')
final_latent = traj_latent[-1]

import torch
with torch.no_grad():
    logits = model(torch.tensor(final_latent))
    pred = logits.argmax(dim=1)
    labels = le.inverse_transform(pred.numpy())

# 统计细胞类型分布
from collections import Counter
print(Counter(labels))
```

### 示例 3: Agent 工具调用

```python
from cytobridge_agent.tools.downstream_analysis_toolkit import DownstreamAnalysisToolkit

# 初始化
toolkit = DownstreamAnalysisToolkit(
    adata_path='trained_model.h5ad',
    output_dir='output/'
)

# 训练分类器
toolkit.train_cell_classifier(label_key='cell_type', epochs=100)

# 生成轨迹
toolkit.generate_trajectory_dataset(
    n_cells=100,
    n_steps=50,
    method='sde'
)

# 扰动分析
toolkit.analyze_gene_perturbation(
    genes=['Klf4', 'Irf8'],
    z_scores=[-2, -1, 0, 1, 2]
)
```

---

## 文件位置

| 文件 | 路径 |
|------|------|
| 核心模块 | `CytoBridge-main/CytoBridge/tl/perturbation.py` |
| Agent 工具 | `cytobridge_agent/tools/downstream_analysis_toolkit.py` |
| 模块导出 | `CytoBridge-main/CytoBridge/tl/__init__.py` |

---

## 参考

- **DeepRUOT**: 原始扰动分析流程来源
- **CytoBridge DynamicalModel**: 速度场、生长率、评分网络
- **SDE 积分**: Euler-Maruyama 方法
