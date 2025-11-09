# 基于全局基准材料的基座模型方案

## 核心思想对比

### 方案 A：用户颜料库正向采样（我之前的方案）

```
用户颜料库（10-20 种）
    ↓
随机采样配方（2-5 种颜料）
    ↓
物理混合（KM 理论）
    ↓
生成训练数据（K/S, Lab, 反射率）
    ↓
训练模型（特定于用户）
```

**特点**：
- ✅ 快速迭代，简单直接
- ✅ 保证可实现性（用户手头有这些颜料）
- ⚠️ 泛化能力有限（限定于用户颜料库）
- ⚠️ 每个用户需要独立训练

---

### 方案 B：全局基准材料 + 基座模型（您的方案）

```
收集全行业基准材料 K/S（500-1000 种）
    ↓
通过凸组合扩展（100 万+ 样本）
    ↓
训练基座模型（理解 K/S 空间通用结构）
    ↓
用户微调（适配特定颜料库）
```

**特点**：
- ✅ 理论完备（覆盖所有可混合颜色）
- ✅ 泛化能力强（理解 K/S 空间本质）
- ✅ 数据可追溯（基于真实材料）
- ✅ 符合物理定律（不是统计黑箱）
- ⚠️ 需要大规模基准数据收集
- ⚠️ 训练成本较高

---

## 您的方案的理论基础

### 物理逻辑链

```
K/S 是物理常数
    ↓
真实材料的 K/S 空间是有限的、可枚举的
    ↓
K/S 混合满足线性性（凸组合）
    ↓
从基准材料出发，可以扩展出完整的"可实现空间"
    ↓
在这个空间上训练，模型学习物理混合规律
    ↓
面对新输入，模型拆分到可用的 K/S 组合
    ↓
通过微调适配具体材料库
```

**这不是统计学习，而是物理约束下的函数逼近！**

---

## 实施方案

### 阶段 1：基准数据收集

#### 1.1 数据来源

```python
# 收集全行业基准 K/S 数据
baseline_materials = {
    # 颜料（Pigments）
    'inorganic_pigments': {
        'titanium_dioxide': ks_data,     # 白色
        'carbon_black': ks_data,         # 黑色
        'iron_oxide_red': ks_data,       # 红色
        # ... ~200 种常见颜料
    },

    # 染料（Dyes）
    'textile_dyes': {
        'reactive_dyes': [...],
        'acid_dyes': [...],
        # ... ~100 种染料
    },

    # 墨水（Inks）
    'printing_inks': {
        'cyan': ks_data,
        'magenta': ks_data,
        'yellow': ks_data,
        'black': ks_data,
        # ... ~50 种墨水
    },

    # 涂料（Coatings）
    'automotive_coatings': [...],
    'architectural_paints': [...],

    # 其他材料
    'cosmetics': [...],
    'food_colorants': [...],
}

# 总计：500-1000 条真实 K/S 曲线
```

#### 1.2 数据标准化

```python
class BaselineMaterialLibrary:
    """基准材料库"""

    def __init__(self):
        self.materials = []
        self.wavelengths = np.arange(400, 701, 10)  # 标准波长

    def add_material(self, name, K, S, metadata=None):
        """
        添加基准材料

        参数:
            name: 材料名称
            K: 吸收系数 (31,)
            S: 散射系数 (31,)
            metadata: 元数据（类别、来源、测量条件等）
        """
        # 验证数据
        assert len(K) == 31, "K 必须有 31 个波长点"
        assert len(S) == 31, "S 必须有 31 个波长点"
        assert np.all(K >= 0), "K 必须非负"
        assert np.all(S >= 0), "S 必须非负"

        # 插值到标准波长（如果需要）
        if metadata and 'wavelengths' in metadata:
            K = np.interp(self.wavelengths, metadata['wavelengths'], K)
            S = np.interp(self.wavelengths, metadata['wavelengths'], S)

        self.materials.append({
            'name': name,
            'K': K,
            'S': S,
            'metadata': metadata or {}
        })

    def get_all_ks(self):
        """获取所有 K/S 数据"""
        return np.array([
            np.concatenate([m['K'], m['S']])  # 拼接为 62 维向量
            for m in self.materials
        ])

    def save(self, path):
        """保存材料库"""
        np.save(path, {
            'materials': self.materials,
            'wavelengths': self.wavelengths
        })
```

#### 1.3 实际数据源

可能的来源：
1. **商业数据库**：
   - X-Rite PantoneLIVE
   - Datacolor ENVISION
   - 各颜料厂商的技术手册

2. **学术文献**：
   - 搜索关键词："Kubelka-Munk", "K/S data", "reflectance spectrum"
   - 数据集论文（如 Munsell 色卡的 K/S）

3. **开放数据**：
   - CIE 数据库
   - 纺织行业标准库

4. **实验室测量**：
   - 购买常见颜料样品
   - 用分光光度计测量反射率
   - 逆推 K/S（从反射率反算）

---

### 阶段 2：物理扩展（凸组合）

#### 2.1 扩展策略

```python
def generate_foundation_training_data(
    baseline_library: BaselineMaterialLibrary,
    n_samples: int = 1000000
):
    """
    从基准材料库生成训练数据

    策略：凸组合（物理可实现）
    """
    print(f"从 {len(baseline_library.materials)} 种基准材料"
          f"生成 {n_samples} 个训练样本...")

    data = []
    baseline_ks = baseline_library.get_all_ks()  # (N, 62)

    for i in range(n_samples):
        # 1. 随机选择材料数量（2-5 种）
        n_components = np.random.randint(2, 6)

        # 2. 随机选择材料
        selected_indices = np.random.choice(
            len(baseline_library.materials),
            n_components,
            replace=False
        )

        # 3. Dirichlet 分布采样浓度（确保和为 1）
        concentrations = np.random.dirichlet(np.ones(n_components))

        # 4. 凸组合（物理混合）
        ks_mixed = np.zeros(62)
        for idx, conc in zip(selected_indices, concentrations):
            ks_mixed += baseline_ks[idx] * conc

        # 5. 计算反射率和 Lab
        K, S = ks_mixed[:31], ks_mixed[31:]
        reflectance = ks_to_reflectance(K, S)
        lab = reflectance_to_lab(reflectance, wavelengths)

        data.append({
            'target_ks': ks_mixed,
            'components': selected_indices,
            'concentrations': concentrations,
            'reflectance': reflectance,
            'lab': lab
        })

        if (i + 1) % 100000 == 0:
            print(f"  已生成 {i+1}/{n_samples} 样本")

    return data
```

#### 2.2 覆盖度验证

```python
def verify_coverage(training_data, test_ks_samples):
    """
    验证训练数据的覆盖度

    方法：
    - 对测试集中的每个 K/S
    - 找到训练集中最近的 k 个邻居
    - 计算插值误差
    """
    from sklearn.neighbors import NearestNeighbors

    # 提取训练集的 K/S
    train_ks = np.array([d['target_ks'] for d in training_data])

    # 构建 KNN
    nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean')
    nbrs.fit(train_ks)

    # 验证
    errors = []
    for test_ks in test_ks_samples:
        distances, indices = nbrs.kneighbors([test_ks])

        # 最近邻平均
        neighbors = train_ks[indices[0]]
        interpolated = np.mean(neighbors, axis=0)

        # 计算误差
        error = np.linalg.norm(test_ks - interpolated)
        errors.append(error)

    print(f"覆盖度统计:")
    print(f"  平均插值误差: {np.mean(errors):.6f}")
    print(f"  中位数误差: {np.median(errors):.6f}")
    print(f"  最大误差: {np.max(errors):.6f}")
    print(f"  95% 分位数: {np.percentile(errors, 95):.6f}")

    # 判断是否足够密集
    threshold = 0.01  # 根据实际需求调整
    coverage_rate = np.mean(np.array(errors) < threshold)

    print(f"  覆盖率（误差 < {threshold}）: {coverage_rate*100:.1f}%")

    return coverage_rate > 0.95  # 95% 以上样本误差小于阈值
```

#### 2.3 智能采样（避免维度诅咒）

```python
def intelligent_sampling(baseline_library, n_samples=1000000):
    """
    智能采样：避免均匀采样 62 维空间

    策略：
    1. 聚类分析找到"代表性"材料组
    2. 重点采样"边界"区域（颜色空间边缘）
    3. 使用重要性采样（常用组合采样更密）
    """
    from sklearn.cluster import KMeans

    baseline_ks = baseline_library.get_all_ks()

    # 1. 聚类（例如分成 20 组）
    kmeans = KMeans(n_clusters=20, random_state=42)
    clusters = kmeans.fit_predict(baseline_ks)

    # 2. 分配采样配额
    data = []
    samples_per_cluster = n_samples // 20

    for cluster_id in range(20):
        cluster_materials = np.where(clusters == cluster_id)[0]

        for _ in range(samples_per_cluster):
            # 在簇内随机采样
            n_components = np.random.randint(2, 6)
            selected = np.random.choice(
                cluster_materials,
                min(n_components, len(cluster_materials)),
                replace=False
            )

            concentrations = np.random.dirichlet(np.ones(len(selected)))

            # 混合
            ks_mixed = np.sum([
                baseline_ks[i] * c
                for i, c in zip(selected, concentrations)
            ], axis=0)

            data.append({
                'target_ks': ks_mixed,
                'components': selected,
                'concentrations': concentrations,
                'cluster': cluster_id
            })

    return data
```

---

### 阶段 3：基座模型训练

#### 3.1 模型架构

```python
class FoundationKSDecompositionModel(nn.Module):
    """
    基座模型：K/S 分解

    输入：目标 K/S 曲线 (62 维)
    输出：基准材料库的浓度 (N 维)
    """

    def __init__(self, n_baseline_materials, hidden_dim=512):
        super().__init__()

        self.n_baseline = n_baseline_materials

        # Encoder: K/S → 潜在表示
        self.encoder = nn.Sequential(
            nn.Linear(62, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden_dim),
            nn.ReLU()
        )

        # Decoder: 潜在表示 → 浓度
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_baseline_materials),
            nn.Softmax(dim=1)  # 确保浓度和为 1
        )

    def forward(self, target_ks, baseline_ks_library):
        """
        参数:
            target_ks: (B, 62) - 目标 K/S
            baseline_ks_library: (N, 62) - 基准材料库

        返回:
            concentrations: (B, N) - 基准材料浓度
            reconstructed_ks: (B, 62) - 重建的 K/S
        """
        # 编码
        latent = self.encoder(target_ks)

        # 解码为浓度
        concentrations = self.decoder(latent)

        # 重建 K/S（物理混合）
        # (B, N) @ (N, 62) = (B, 62)
        reconstructed_ks = torch.matmul(concentrations, baseline_ks_library)

        return concentrations, reconstructed_ks

    def loss_function(self, target_ks, reconstructed_ks,
                     concentrations, lambda_sparsity=0.01):
        """
        损失函数

        1. K/S 重建损失（主要）
        2. 稀疏性损失（倾向少用材料）
        3. 物理约束损失（可选）
        """
        # 1. 重建损失
        reconstruction_loss = F.mse_loss(reconstructed_ks, target_ks)

        # 2. 稀疏性损失（L1 正则）
        sparsity_loss = torch.mean(torch.abs(concentrations))

        # 3. 总损失
        total_loss = (
            reconstruction_loss +
            lambda_sparsity * sparsity_loss
        )

        return total_loss, {
            'reconstruction': reconstruction_loss.item(),
            'sparsity': sparsity_loss.item()
        }
```

#### 3.2 训练流程

```python
def train_foundation_model(
    baseline_library,
    training_data,
    validation_data,
    epochs=100,
    batch_size=512,
    lr=1e-3
):
    """训练基座模型"""

    # 创建模型
    n_baseline = len(baseline_library.materials)
    model = FoundationKSDecompositionModel(n_baseline)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 基准材料库（固定）
    baseline_ks = torch.FloatTensor(baseline_library.get_all_ks())

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # 训练
    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in DataLoader(training_data, batch_size=batch_size, shuffle=True):
            target_ks = batch['target_ks']

            # 前向
            concentrations, reconstructed_ks = model(target_ks, baseline_ks)
            loss, loss_dict = model.loss_function(
                target_ks, reconstructed_ks, concentrations
            )

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in DataLoader(validation_data, batch_size=batch_size):
                target_ks = batch['target_ks']
                concentrations, reconstructed_ks = model(target_ks, baseline_ks)
                loss, _ = model.loss_function(target_ks, reconstructed_ks, concentrations)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}: "
              f"Train Loss = {np.mean(train_losses):.6f}, "
              f"Val Loss = {avg_val_loss:.6f}")

    return model
```

---

### 阶段 4：用户微调

```python
def finetune_for_user(
    foundation_model,
    baseline_library,
    user_pigment_library,
    epochs=20
):
    """
    针对用户颜料库微调

    策略：
    1. 找到用户颜料在基准库中的对应关系
    2. 冻结 Encoder，只微调 Decoder
    3. 添加"材料映射层"
    """

    # 1. 材料映射：用户颜料 → 基准材料
    mapping = map_user_to_baseline(user_pigment_library, baseline_library)

    # 2. 创建微调模型
    class FinetunedModel(nn.Module):
        def __init__(self, foundation, user_library_size):
            super().__init__()
            self.encoder = foundation.encoder  # 冻结

            # 新的解码器：适配用户颜料库
            self.user_decoder = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, user_library_size),
                nn.Softmax(dim=1)
            )

        def forward(self, target_ks, user_ks_library):
            latent = self.encoder(target_ks)
            concentrations = self.user_decoder(latent)
            reconstructed_ks = torch.matmul(concentrations, user_ks_library)
            return concentrations, reconstructed_ks

    # 3. 冻结 Encoder
    for param in foundation_model.encoder.parameters():
        param.requires_grad = False

    # 4. 微调
    finetuned_model = FinetunedModel(
        foundation_model,
        len(user_pigment_library)
    )

    # ... 训练流程类似

    return finetuned_model
```

---

## 关键挑战与解决方案

### 挑战 1：线性性偏差

**问题**：K/S 混合的线性性是近似的

**解决方案**：
```python
# 方案 A：数据增强（引入非线性修正）
def add_nonlinear_correction(K_mix, S_mix):
    """
    经验修正（基于实验数据拟合）
    """
    # 例如：粒径效应修正
    particle_size_factor = ...
    K_corrected = K_mix * (1 + particle_size_factor)
    return K_corrected, S_mix

# 方案 B：模型学习非线性
class NonlinearKSModel(nn.Module):
    """在线性基础上学习非线性残差"""
    def forward(self, target_ks, baseline_ks):
        # 线性组合
        concentrations = self.decoder(...)
        ks_linear = torch.matmul(concentrations, baseline_ks)

        # 非线性修正
        residual = self.correction_network(ks_linear)
        ks_corrected = ks_linear + residual

        return concentrations, ks_corrected
```

### 挑战 2：数据获取困难

**解决方案**：
1. **分阶段收集**：
   - 第一阶段：100 种核心材料（CMYK + 常见颜料）
   - 第二阶段：扩展到 500 种
   - 第三阶段：行业合作获取更多

2. **数据共享**：
   - 建立开源 K/S 数据库
   - 鼓励用户贡献数据（去隐私化）

3. **合成数据验证**：
   - 用物理模拟生成候选 K/S
   - 用少量真实数据验证

### 挑战 3：计算效率

**解决方案**：
```python
# 稀疏化技术
class SparseKSDecomposition(nn.Module):
    """
    强制稀疏输出：只用 top-k 个材料
    """
    def forward(self, target_ks, baseline_ks, k=5):
        concentrations_full = self.decoder(...)

        # Top-K 选择
        topk_values, topk_indices = torch.topk(concentrations_full, k)

        # 重新归一化
        concentrations_sparse = torch.zeros_like(concentrations_full)
        concentrations_sparse.scatter_(
            1, topk_indices,
            F.softmax(topk_values, dim=1)
        )

        return concentrations_sparse
```

---

## 理论保证

### 定理（非正式）

**如果**：
1. 基准材料库覆盖了 K/S 空间的"骨架"
2. 训练数据通过凸组合充分扩展
3. 采样足够密集（平均最近邻距离 < ε）

**则**：
- 对任意"可混合"的目标 K/S
- 存在基准材料的组合，使得误差 < δ
- 其中 δ = O(ε)（误差与采样密度线性相关）

**证明思路**（略）：
- K/S 空间是凸空间
- 凸组合的闭包可以逼近任意点
- 神经网络是万能逼近器

---

## 实施建议

### 最小可行产品（MVP）

```
第一阶段（1-2 个月）：
1. 收集 100 种核心材料的 K/S 数据
2. 生成 10 万个训练样本（凸组合）
3. 训练小型基座模型（验证可行性）
4. 在真实配方上测试

第二阶段（3-6 个月）：
1. 扩展到 500 种材料
2. 生成 100 万样本
3. 训练大型基座模型
4. 实现用户微调流程
5. 覆盖度验证

第三阶段（6-12 个月）：
1. 建立开源数据库
2. 行业合作收集数据
3. 非线性修正
4. 生产部署
```

---

## 总结

您的方案**在理论上非常优雅，在工程上可行**！

**核心优势**：
- 物理基础扎实（不是黑箱）
- 泛化能力强（理解 K/S 本质）
- 数据可追溯（基于真实材料）

**实施路径**：
1. MVP：100 材料 + 10 万样本
2. 扩展：500 材料 + 100 万样本
3. 优化：非线性修正 + 稀疏化

这是一个**值得深入研究**的方向，有很大的学术和商业价值！
