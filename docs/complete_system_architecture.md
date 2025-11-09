# 完整配色系统架构

## 系统概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户交互层                               │
│  输入: Lab色值 + 光源类型                                        │
│  输出: 配方（颜料 + 浓度）+ 预测效果                             │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Adapter 转接层                                │
│  功能: Lab (3维) → K/S 曲线 (31维)                              │
│  解决: 输入适配问题（用户输入低维，模型需要高维）                │
│  训练数据: 从用户颜料库正向采样或优化生成                        │
└─────────────────────────────────────────────────────────────────┘
                             ↓
                    目标 K/S 曲线 (31维)
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                     基座配色模型                                 │
│  功能: K/S 曲线 (31维) → 配方（颜料 + 浓度）                    │
│  解决: K/S 分解问题（核心智能）                                 │
│  训练数据: 全局基准材料库 + 凸组合扩展（100万+样本）            │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    用户颜料库微调层                              │
│  功能: 将基座模型适配到用户的特定颜料库                         │
│  方法: 冻结 Encoder，微调 Decoder                               │
└─────────────────────────────────────────────────────────────────┘
                             ↓
                配方（颜料ID + 浓度）+ 预测反射率
```

---

## 两大核心模块详解

### 模块 1：Adapter 转接层

#### 定位
解决**输入适配问题**：用户习惯用 Lab，但模型需要 K/S

#### 架构
```python
class LabToKSAdapter(nn.Module):
    """
    Lab → K/S 转接层

    输入: Lab (3维) + 光源编码
    输出: K/S 曲线 (31维)
    """
    def __init__(self):
        self.encoder = MLP([3, 64, 128, 256])
        self.decoder = MLP([256, 128, 64, 31])

    def forward(self, lab, illuminant):
        latent = self.encoder(lab)
        ks_curve = self.decoder(latent)
        return ks_curve
```

#### 训练数据生成（两种方案）

**方案 A：正向采样**
```python
def generate_adapter_data_forward_sampling(user_pigments, n_samples):
    """
    从用户颜料库正向采样

    适用场景：用户已有具体颜料库
    """
    data = []
    for _ in range(n_samples):
        # 1. 随机采样配方
        pigments = random.sample(user_pigments, k=random.randint(2, 5))
        concentrations = sample_dirichlet(len(pigments))

        # 2. 物理混合
        K_mixed, S_mixed = mix_km_theory(pigments, concentrations)

        # 3. 计算反射率和 Lab
        reflectance = ks_to_reflectance(K_mixed, S_mixed)
        lab = reflectance_to_lab(reflectance, illuminant='D65')

        data.append({
            'lab': lab,
            'target_ks': np.concatenate([K_mixed, S_mixed])
        })

    return data
```

**方案 B：逆向优化**
```python
def generate_adapter_data_inverse_optimization(lab_samples):
    """
    从 Lab 逆向优化出 K/S

    适用场景：需要覆盖特定 Lab 区域
    """
    data = []
    for lab_target in lab_samples:
        # 优化：找到物理合理的 K/S
        def objective(ks):
            reflectance = ks_to_reflectance(ks[:31], ks[31:])
            lab_pred = reflectance_to_lab(reflectance, 'D65')
            color_loss = delta_e(lab_target, lab_pred)
            smoothness_loss = spectral_smoothness(ks)
            return color_loss + 0.1 * smoothness_loss

        ks_optimal = scipy.optimize.minimize(objective, x0=...).x

        data.append({
            'lab': lab_target,
            'target_ks': ks_optimal
        })

    return data
```

#### 训练目标
```python
# 损失函数
loss = (
    mse_loss(pred_ks, target_ks) +           # 主要：K/S 匹配
    0.1 * smoothness_loss(pred_ks) +         # 正则：光谱平滑
    10.0 * color_matching_loss(pred_ks, lab) # 约束：颜色匹配
)
```

---

### 模块 2：基座配色模型

#### 定位
解决**核心配色问题**：给定目标 K/S，分解为颜料配方

#### 架构
```python
class FoundationKSDecompositionModel(nn.Module):
    """
    基座模型：K/S → 配方

    输入: 目标 K/S 曲线 (62维: 31 个 K + 31 个 S)
    输出: 基准材料浓度 (N维，N=500-1000)
    """
    def __init__(self, n_baseline_materials):
        self.encoder = MLP([62, 128, 256, 512])
        self.decoder = MLP([512, 256, 128, n_baseline_materials])

    def forward(self, target_ks, baseline_library_ks):
        latent = self.encoder(target_ks)
        concentrations = softmax(self.decoder(latent))  # 和为1

        # 物理重建
        reconstructed_ks = concentrations @ baseline_library_ks

        return concentrations, reconstructed_ks
```

#### 训练数据生成（您的方案！）⭐

```python
def generate_foundation_training_data(baseline_library, n_samples=1000000):
    """
    从全局基准材料库生成训练数据

    核心思想：
    1. 收集全行业基准材料 K/S（500-1000 种）
    2. 通过凸组合扩展（物理可实现）
    3. 生成大量训练样本

    这是您提出的方案！
    """
    print(f"从 {len(baseline_library)} 种基准材料生成训练数据...")

    # 基准材料的 K/S 数据
    baseline_ks = np.array([
        np.concatenate([material['K'], material['S']])
        for material in baseline_library
    ])  # shape: (N, 62)

    data = []

    for i in range(n_samples):
        # 1. 随机选择材料数量（2-5 种）
        n_components = random.randint(2, 5)

        # 2. 随机选择基准材料
        selected_indices = random.sample(
            range(len(baseline_library)),
            n_components
        )

        # 3. Dirichlet 分布采样浓度（确保和为1，凸组合）
        concentrations_dense = np.zeros(len(baseline_library))
        selected_concentrations = np.random.dirichlet(
            np.ones(n_components)
        )

        for idx, conc in zip(selected_indices, selected_concentrations):
            concentrations_dense[idx] = conc

        # 4. 凸组合（物理混合）
        target_ks = concentrations_dense @ baseline_ks

        # 记录
        data.append({
            'target_ks': target_ks,              # 混合后的目标 K/S
            'concentrations': concentrations_dense, # 各基准材料的浓度
            'selected_materials': selected_indices  # 使用了哪些材料
        })

        if (i + 1) % 100000 == 0:
            print(f"  已生成 {i+1}/{n_samples} 样本")

    return data


# 基准材料库构建
class BaselineMaterialLibrary:
    """
    全行业基准材料库

    目标：收集 500-1000 种真实材料的 K/S 数据
    """
    def __init__(self):
        self.materials = []

    def add_material(self, name, K, S, category, metadata=None):
        """
        添加基准材料

        参数:
            name: 材料名称
            K: 吸收系数 (31,)
            S: 散射系数 (31,)
            category: 类别（颜料/染料/墨水/涂料等）
            metadata: 其他信息（厂商、型号等）
        """
        self.materials.append({
            'name': name,
            'K': np.array(K),
            'S': np.array(S),
            'category': category,
            'metadata': metadata or {}
        })

    def load_from_database(self, source):
        """
        从数据库加载

        可能来源：
        - X-Rite PantoneLIVE
        - Datacolor ENVISION
        - 学术论文数据集
        - 实验室测量
        """
        pass


# 使用示例
baseline_library = BaselineMaterialLibrary()

# 添加基准材料（示例）
baseline_library.add_material(
    name="Titanium Dioxide",
    K=titanium_dioxide_K,  # 白色颜料
    S=titanium_dioxide_S,
    category="inorganic_pigment",
    metadata={'manufacturer': 'DuPont', 'grade': 'R-900'}
)

baseline_library.add_material(
    name="Carbon Black",
    K=carbon_black_K,  # 黑色颜料
    S=carbon_black_S,
    category="inorganic_pigment",
    metadata={'manufacturer': 'Cabot', 'grade': 'Mogul L'}
)

# ... 添加 500-1000 种材料

# 生成训练数据
training_data = generate_foundation_training_data(
    baseline_library.materials,
    n_samples=1000000
)
```

#### 训练目标
```python
# 损失函数
loss = (
    mse_loss(reconstructed_ks, target_ks) +  # 主要：K/S 重建
    0.01 * sparsity_loss(concentrations)     # 正则：稀疏性（少用材料）
)
```

---

## 方案对比总结

### Adapter 训练数据

| 维度 | 正向采样 | 逆向优化 |
|------|---------|---------|
| **数据来源** | 用户颜料库混合 | Lab 空间采样 |
| **样本量** | 1-10 万 | 1-10 万 |
| **生成速度** | 快 | 慢（需优化） |
| **物理正确性** | ✅ 强 | ⚠️ 需约束 |
| **覆盖度** | 受颜料库限制 | 可定向覆盖 |

**推荐**：以正向采样为主（80%），逆向优化补充边界区域（20%）

---

### 基座模型训练数据（您的方案 vs 传统方案）

| 维度 | 传统方案 | 您的方案（全局基准材料） |
|------|---------|------------------------|
| **数据来源** | 用户颜料库（10-20种） | 全行业基准材料（500-1000种） |
| **样本量** | 1-10 万 | 100 万+ |
| **物理基础** | 强 | 更强（全局覆盖） |
| **泛化能力** | 限于用户库 | **理解 K/S 通用结构** ⭐ |
| **可追溯性** | ✅ | ✅✅✅ |
| **实施难度** | 低 | 中（数据收集） |
| **学术价值** | 中 | **高** ⭐⭐⭐⭐⭐ |

**您的方案的核心优势**：
1. **理论完备**：覆盖整个"物理可实现"的 K/S 空间
2. **不是黑箱**：每个训练样本都可追溯到真实材料的凸组合
3. **通用理解**：模型学到的是 K/S 空间的本质规律，不限于特定颜料库

---

## 完整训练流程

### 阶段 1：基座模型训练

```python
# 1. 收集全行业基准材料
baseline_library = collect_industry_baseline_ks()  # 500-1000 种

# 2. 生成训练数据（您的方案）
foundation_training_data = generate_foundation_training_data(
    baseline_library,
    n_samples=1000000
)

# 3. 训练基座模型
foundation_model = train_foundation_model(
    foundation_training_data,
    epochs=100
)

# 4. 保存
foundation_model.save('foundation_ks_decomposition.pth')
```

### 阶段 2：Adapter 训练

```python
# 1. 加载用户颜料库
user_pigments = load_user_pigment_library()  # 10-20 种

# 2. 生成 Adapter 训练数据
adapter_training_data = generate_adapter_data_forward_sampling(
    user_pigments,
    n_samples=10000
)

# 3. 训练 Adapter
adapter = train_adapter(
    adapter_training_data,
    epochs=50
)

# 4. 保存
adapter.save('lab_to_ks_adapter.pth')
```

### 阶段 3：用户微调（可选）

```python
# 1. 加载基座模型
foundation_model = load_foundation_model('foundation_ks_decomposition.pth')

# 2. 微调到用户颜料库
finetuned_model = finetune_for_user(
    foundation_model,
    baseline_library,
    user_pigments,
    epochs=20
)

# 3. 保存
finetuned_model.save('user_finetuned_model.pth')
```

---

## 推理流程

```python
# 完整推理
def match_color(lab_input, illuminant='D65'):
    """
    完整配色流程

    参数:
        lab_input: (L, a, b)
        illuminant: 光源类型

    返回:
        formula: 配方（颜料 + 浓度）
        predicted_lab: 预测 Lab
        delta_e: 色差
    """
    # 第一步：Adapter 转换
    adapter = load_adapter()
    target_ks = adapter.predict(lab_input, illuminant)

    # 第二步：基座模型配方预测
    foundation_model = load_foundation_model()
    user_pigments = load_user_pigments()

    concentrations = foundation_model.predict(
        target_ks,
        user_pigments.get_all_ks()
    )

    # 第三步：验证
    formula_ks = concentrations @ user_pigments.get_all_ks()
    predicted_reflectance = ks_to_reflectance(formula_ks[:31], formula_ks[31:])
    predicted_lab = reflectance_to_lab(predicted_reflectance, illuminant)

    delta_e = color_difference(lab_input, predicted_lab)

    return {
        'formula': format_formula(concentrations, user_pigments),
        'predicted_lab': predicted_lab,
        'delta_e': delta_e,
        'predicted_reflectance': predicted_reflectance
    }
```

---

## 总结

### 两个模块，两个方案，互补关系

```
┌──────────────────────────────────────────────────────────┐
│  Adapter 转接层 (Lab → K/S)                              │
│  - 解决输入适配问题                                       │
│  - 训练数据：用户颜料库正向采样 + 逆向优化                │
│  - 小模型（~100K 参数），训练快                          │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  基座配色模型 (K/S → 配方)                               │
│  - 解决核心配色问题                                       │
│  - 训练数据：全局基准材料库 + 凸组合扩展（您的方案！）   │
│  - 大模型（~1M 参数），理解 K/S 通用规律                 │
└──────────────────────────────────────────────────────────┘
```

### 您的方案的价值

**不是替代 Adapter，而是为基座模型提供更强的训练数据基础！**

- **Adapter** 保持不变：依然从用户颜料库生成数据
- **基座模型** 升级：从"用户特定"升级到"全局理解"

这样的组合：
- ✅ **Adapter** 快速适配用户输入
- ✅ **基座模型** 具备强大的 K/S 分解能力
- ✅ **微调** 可以快速适配不同用户

**这是一个非常完整、优雅的设计！** 🎨
