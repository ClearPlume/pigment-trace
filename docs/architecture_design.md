# 配色系统架构设计

## 核心思路

基于物理正确的 K/S 曲线构建核心模型，通过转接层处理实际应用中的低维输入。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户输入层                               │
│  输入: Lab色值 + 光源类型 + 约束条件                          │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   前置转接层 (Adapter)                        │
│  功能: Lab (3维) → K/S曲线 (31维)                            │
│  方法: 约束优化 或 神经网络 + 正则化                          │
│  约束:                                                        │
│    - 光谱平滑度 (避免不自然振荡)                              │
│    - 反射率边界 [0, 1]                                        │
│    - 最小化先验距离 (如灰度光谱)                              │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   核心配色模型 (Foundation)                   │
│  输入: 目标 K/S 曲线 (31维)                                  │
│  输出: 配方 K/S 组合 (N个颜料 × 31维 + 浓度)                 │
│  训练:                                                        │
│    - 基座: 物理合成数据 (通用颜料库)                          │
│    - 微调: 用户特定颜料库                                     │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   后置验证层 (Validator)                      │
│  功能: 验证配方是否满足要求                                   │
│  计算:                                                        │
│    - 配方 K/S → 反射率曲线 R(λ)                              │
│    - R(λ) → Lab (在指定光源下)                               │
│    - 计算色差 ΔE                                             │
│  可选: 如果 ΔE > 阈值，反馈优化                               │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                      输出结果                                 │
│  - 颜料配方 (颜料名称 + 浓度)                                 │
│  - 预测反射率曲线                                             │
│  - 预测Lab值 + 色差                                          │
│  - 成本估算                                                   │
└─────────────────────────────────────────────────────────────┘
```

## 详细设计

### 1. 前置转接层

#### 挑战
- **一对多映射**: 同一个 Lab 对应无穷多条 K/S 曲线
- **需要约束**: 必须加入额外条件选择"合理"的 K/S

#### 解决方案A: 约束优化

```python
def lab_to_ks_optimization(
    lab_target: tuple[float, float, float],
    illuminant: str = "D65",
    constraints: dict = None
) -> np.ndarray:
    """
    通过优化将 Lab 转换为 K/S 曲线

    优化目标:
        minimize: color_loss + λ1·smoothness + λ2·prior_loss
        subject to: 0 ≤ R(λ) ≤ 1
    """
    wavelengths = np.arange(400, 701, 10)

    def objective(reflectance):
        # 1. 颜色匹配损失
        lab_pred = spectrum_to_lab(reflectance, wavelengths, illuminant)
        color_loss = delta_e(lab_target, lab_pred) ** 2

        # 2. 光谱平滑损失 (一阶导数)
        smoothness = np.sum(np.diff(reflectance) ** 2)

        # 3. 先验损失 (倾向于平坦光谱)
        L_target, _, _ = lab_target
        neutral_reflectance = L_target / 100.0
        prior_loss = np.sum((reflectance - neutral_reflectance) ** 2)

        return color_loss + 0.1 * smoothness + 0.01 * prior_loss

    # 初始猜测: 根据明度估计平坦光谱
    x0 = np.ones(31) * (lab_target[0] / 100.0)

    # 优化
    result = scipy.optimize.minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=[(0.01, 0.99)] * 31
    )

    return result.x
```

**优点**:
- 物理可解释
- 不需要训练
- 可以灵活调整约束

**缺点**:
- 计算较慢 (每次推理都要优化)
- 需要手动调整超参数 λ

#### 解决方案B: 神经网络转接层

```python
class LabToSpectrumAdapter(nn.Module):
    """
    可学习的 Lab → K/S 转接层
    """
    def __init__(self, n_wavelengths=31):
        super().__init__()

        # 编码器: Lab (3维) → 潜在空间
        self.lab_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # 光源编码
        self.illuminant_embedding = nn.Embedding(
            num_embeddings=5,  # D65, A, F11, etc.
            embedding_dim=16
        )

        # 解码器: 潜在空间 → K/S 曲线
        self.decoder = nn.Sequential(
            nn.Linear(128 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_wavelengths),
            nn.Sigmoid()  # 确保输出在 [0, 1]
        )

    def forward(self, lab, illuminant_id):
        """
        参数:
            lab: (B, 3) - Lab 色值
            illuminant_id: (B,) - 光源ID
        返回:
            reflectance: (B, 31) - 反射率曲线
        """
        lab_features = self.lab_encoder(lab)
        illum_features = self.illuminant_embedding(illuminant_id)

        combined = torch.cat([lab_features, illum_features], dim=1)
        reflectance = self.decoder(combined)

        return reflectance


class AdapterLoss(nn.Module):
    """
    转接层的多任务损失函数
    """
    def __init__(self, lambda_smooth=0.1, lambda_physics=0.05):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_physics = lambda_physics

    def forward(self, reflectance_pred, lab_target, illuminant):
        """
        参数:
            reflectance_pred: (B, 31) - 预测的反射率
            lab_target: (B, 3) - 目标 Lab
            illuminant: 光源信息
        """
        # 1. 颜色匹配损失
        lab_pred = reflectance_to_lab_batch(
            reflectance_pred, illuminant
        )  # 需要实现可微分版本
        color_loss = torch.mean(
            (lab_pred - lab_target).pow(2).sum(dim=1).sqrt()
        )

        # 2. 光谱平滑损失
        diff = reflectance_pred[:, 1:] - reflectance_pred[:, :-1]
        smoothness_loss = torch.mean(diff.pow(2))

        # 3. 物理约束损失 (软约束，确保在 [0,1] 内)
        physics_loss = torch.mean(
            F.relu(-reflectance_pred) +  # 惩罚 < 0
            F.relu(reflectance_pred - 1)  # 惩罚 > 1
        )

        total_loss = (
            color_loss +
            self.lambda_smooth * smoothness_loss +
            self.lambda_physics * physics_loss
        )

        return total_loss, {
            'color': color_loss.item(),
            'smoothness': smoothness_loss.item(),
            'physics': physics_loss.item()
        }
```

**训练数据生成**:
```python
def generate_adapter_training_data(n_samples=100000):
    """
    生成转接层训练数据

    策略:
    1. 从物理合理的光谱开始
    2. 计算对应的 Lab
    3. 构成训练对 (Lab, Spectrum)
    """
    data = []

    # 方法1: 从真实颜料配方生成
    for _ in range(n_samples // 2):
        # 随机采样颜料组合和浓度
        pigments = sample_pigments(n=3)
        concentrations = sample_concentrations()

        # 计算混合 K/S
        ks_mixed = mix_pigments_ks(pigments, concentrations)
        reflectance = ks_to_reflectance(ks_mixed)

        # 计算各光源下的 Lab
        for illuminant in ['D65', 'A', 'F11']:
            lab = spectrum_to_lab(reflectance, illuminant)
            data.append({
                'lab': lab,
                'illuminant': illuminant,
                'reflectance': reflectance
            })

    # 方法2: 直接采样光谱空间（确保多样性）
    for _ in range(n_samples // 2):
        # 生成平滑的随机光谱
        reflectance = generate_smooth_spectrum()

        for illuminant in ['D65', 'A', 'F11']:
            lab = spectrum_to_lab(reflectance, illuminant)
            data.append({
                'lab': lab,
                'illuminant': illuminant,
                'reflectance': reflectance
            })

    return data


def generate_smooth_spectrum():
    """生成物理合理的平滑光谱"""
    # 使用低频成分构建
    n_components = 5  # 少量基函数确保平滑
    wavelengths = np.arange(400, 701, 10)

    spectrum = np.zeros(len(wavelengths))
    for i in range(n_components):
        amplitude = np.random.uniform(0.1, 0.3)
        frequency = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)

        spectrum += amplitude * np.sin(
            2 * np.pi * frequency * np.linspace(0, 1, len(wavelengths)) + phase
        )

    # 归一化到 [0, 1]
    spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
    spectrum = np.clip(spectrum * 0.9 + 0.05, 0, 1)  # 留边界

    return spectrum
```

**优点**:
- 推理快速
- 端到端可微
- 可以从数据学习"合理"的光谱

**缺点**:
- 需要训练数据
- 可能产生物理上不太自然的光谱（需要强正则化）

#### 推荐: 混合方案

```python
class HybridLabToKSAdapter:
    """
    混合转接层:
    - 训练时: 用优化生成高质量数据
    - 推理时: 神经网络快速预测 + 可选精修
    """
    def __init__(self):
        self.neural_adapter = LabToSpectrumAdapter()
        self.optimizer_config = {...}

    def generate_training_data(self, n_samples):
        """离线生成训练数据"""
        data = []
        for lab, illuminant in sample_lab_space(n_samples):
            # 用优化得到高质量 K/S
            ks = lab_to_ks_optimization(
                lab, illuminant, self.optimizer_config
            )
            data.append((lab, illuminant, ks))
        return data

    def train(self, data, epochs):
        """训练神经网络模仿优化器"""
        # 标准监督学习
        for epoch in range(epochs):
            for lab, illuminant, ks_target in data:
                ks_pred = self.neural_adapter(lab, illuminant)
                loss = mse_loss(ks_pred, ks_target)
                loss.backward()
                optimizer.step()

    def predict(self, lab, illuminant, refine=False):
        """推理"""
        # 快速预测
        ks_init = self.neural_adapter(lab, illuminant)

        if not refine:
            return ks_init

        # 可选: 精修 (从神经网络预测开始优化)
        ks_refined = lab_to_ks_optimization(
            lab, illuminant,
            x0=ks_init,  # 用神经网络输出初始化
            max_iter=50  # 少量迭代即可
        )
        return ks_refined
```

### 2. 核心配色模型

```python
class PigmentFormulationModel(nn.Module):
    """
    核心模型: K/S → 配方
    """
    def __init__(
        self,
        n_wavelengths=31,
        max_pigments=10,
        hidden_dim=256
    ):
        super().__init__()

        # 编码器: 目标 K/S → 潜在表示
        self.target_encoder = nn.Sequential(
            nn.Linear(n_wavelengths, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

        # 解码器: 潜在表示 → 颜料选择 + 浓度
        self.pigment_selector = nn.Linear(hidden_dim, max_pigments)
        self.concentration_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_pigments),
            nn.Softmax(dim=1)  # 浓度和为1
        )

    def forward(self, target_ks, pigment_library_ks):
        """
        参数:
            target_ks: (B, 31) - 目标 K/S 曲线
            pigment_library_ks: (P, 31) - 颜料库的 K/S
        返回:
            concentrations: (B, P) - 各颜料浓度
        """
        features = self.target_encoder(target_ks)

        # 预测浓度
        concentrations = self.concentration_predictor(features)

        return concentrations

    def compute_loss(self, concentrations, target_ks, pigment_library_ks):
        """
        损失函数
        """
        # 1. 根据浓度混合颜料
        # concentrations: (B, P)
        # pigment_library_ks: (P, 31)
        # 混合: (B, P) @ (P, 31) = (B, 31)
        predicted_ks = torch.matmul(concentrations, pigment_library_ks)

        # 2. K/S 曲线匹配损失
        ks_loss = F.mse_loss(predicted_ks, target_ks)

        # 3. 稀疏性损失 (倾向少用颜料)
        sparsity_loss = torch.mean(torch.sum(concentrations > 0.01, dim=1))

        # 4. 浓度正则化
        concentration_reg = torch.mean(concentrations.pow(2))

        total_loss = ks_loss + 0.01 * sparsity_loss + 0.001 * concentration_reg

        return total_loss
```

### 3. 训练策略

#### 基座模型训练

```python
# 1. 生成大量物理合成数据
def generate_foundation_training_data(n_samples=1000000):
    """
    使用通用颜料库生成训练数据
    """
    # 通用颜料库 (如 40 种基础颜料)
    pigment_library = load_generic_pigment_library()

    data = []
    for _ in range(n_samples):
        # 随机采样配方
        n_pigments = np.random.randint(2, 6)
        selected_pigments = np.random.choice(
            len(pigment_library), n_pigments, replace=False
        )
        concentrations = sample_dirichlet_concentration(n_pigments)

        # 计算目标 K/S (物理混合)
        target_ks = mix_pigments(
            [pigment_library[i] for i in selected_pigments],
            concentrations
        )

        # 记录
        data.append({
            'target_ks': target_ks,
            'pigment_ids': selected_pigments,
            'concentrations': concentrations
        })

    return data

# 2. 训练基座
foundation_model = PigmentFormulationModel(
    max_pigments=len(generic_pigment_library)
)
train(foundation_model, foundation_data, epochs=100)
```

#### 用户微调

```python
def finetune_for_user(
    foundation_model,
    user_pigment_library,
    user_data=None,
    epochs=20
):
    """
    针对用户特定颜料库微调

    参数:
        foundation_model: 预训练基座
        user_pigment_library: 用户的颜料库 (可能只有10-20种)
        user_data: 用户提供的配方数据 (可选)
        epochs: 微调轮数
    """
    # 1. 如果用户有数据,直接用
    if user_data:
        finetune_data = user_data
    else:
        # 2. 否则,生成合成数据
        finetune_data = generate_synthetic_data(user_pigment_library)

    # 3. 微调 (可以冻结部分层)
    # 只微调最后几层
    for param in foundation_model.target_encoder.parameters():
        param.requires_grad = False

    optimizer = Adam(
        filter(lambda p: p.requires_grad, foundation_model.parameters()),
        lr=1e-4
    )

    train(foundation_model, finetune_data, epochs, optimizer)

    return foundation_model
```

## 完整推理流程

```python
class ColorMatchingSystem:
    """
    完整的配色系统
    """
    def __init__(
        self,
        adapter,
        formulation_model,
        pigment_library
    ):
        self.adapter = adapter  # 前置转接层
        self.model = formulation_model  # 核心模型
        self.pigment_library = pigment_library  # 颜料库

    def match_color(
        self,
        lab_target: tuple,
        illuminant: str = 'D65',
        constraints: dict = None
    ):
        """
        完整的配色流程

        返回:
            formula: 配方 (颜料 + 浓度)
            predicted_lab: 预测的 Lab
            delta_e: 色差
        """
        # 1. 前置转接: Lab → K/S
        target_ks = self.adapter.predict(
            lab_target, illuminant, refine=True
        )

        # 2. 核心模型: K/S → 配方
        concentrations = self.model(
            target_ks,
            self.pigment_library.get_all_ks()
        )

        # 3. 后置验证: 计算实际颜色
        formula_ks = mix_pigments(
            self.pigment_library.get_selected(concentrations),
            concentrations
        )
        reflectance = ks_to_reflectance(formula_ks)
        predicted_lab = spectrum_to_lab(reflectance, illuminant)

        delta_e = color_difference(lab_target, predicted_lab)

        # 4. 如果色差太大,可以迭代优化
        if delta_e > 2.0:
            concentrations = self.refine_formula(
                concentrations, lab_target, illuminant
            )

        # 5. 返回结果
        formula = self.format_formula(concentrations)

        return {
            'formula': formula,
            'predicted_lab': predicted_lab,
            'predicted_spectrum': reflectance,
            'delta_e': delta_e,
            'cost': self.estimate_cost(concentrations)
        }

    def refine_formula(self, initial_conc, lab_target, illuminant):
        """
        精修配方 (可选)
        """
        # 使用梯度下降或优化算法微调浓度
        pass
```

## 总结

您的架构设计**完全成立**,关键点是:

### ✅ 可行性
1. **核心模型基于物理**: K/S → 配方是良定问题
2. **转接层解决实用性**: Lab → K/S 通过约束优化/神经网络解决
3. **分层设计灵活**: 可以独立优化各层

### 🎯 推荐方案
1. **前置转接**: 混合方案 (训练时优化,推理时神经网络)
2. **核心模型**: 基座 + 微调
3. **后置验证**: 物理计算 + 可选迭代

### 📊 预期效果
- **色差**: ΔE < 1.0 (商用标准)
- **速度**: <100ms (神经网络推理)
- **泛化**: 基座模型 + 少量微调即可适应新颜料库

这个架构在理论和工程上都是可行的! 🎨
