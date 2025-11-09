"""
Lab to K/S 转接层实现

将 Lab 色值转换为反射率曲线（K/S），解决一对多映射问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabToKSAdapter(nn.Module):
    """
    Lab → K/S 曲线转接层

    架构：
        Lab (3维) → Encoder → 潜在空间 (256维) → Decoder → 反射率 (31维)
    """

    def __init__(self, n_wavelengths=31, hidden_dim=256):
        super().__init__()

        self.n_wavelengths = n_wavelengths

        # Encoder: Lab → 潜在空间
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )

        # Decoder: 潜在空间 → 反射率曲线
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_wavelengths),
            nn.Sigmoid()  # 确保输出在 [0, 1]
        )

    def forward(self, lab):
        """
        参数:
            lab: (B, 3) - Lab 色值

        返回:
            reflectance: (B, 31) - 反射率曲线
        """
        latent = self.encoder(lab)
        reflectance = self.decoder(latent)
        return reflectance

    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdapterLoss(nn.Module):
    """
    转接层的多任务损失函数
    """

    def __init__(
        self,
        lambda_smooth=0.1,
        lambda_physics=0.0,  # 使用 Sigmoid 时可设为 0
        use_auto_balance=False
    ):
        super().__init__()

        self.lambda_smooth = lambda_smooth
        self.lambda_physics = lambda_physics
        self.use_auto_balance = use_auto_balance

        if use_auto_balance:
            # 可学习的损失权重
            self.log_sigma_color = nn.Parameter(torch.zeros(1))
            self.log_sigma_smooth = nn.Parameter(torch.zeros(1))

    def forward(self, pred_reflectance, target_reflectance, lab_target):
        """
        参数:
            pred_reflectance: (B, 31) - 预测的反射率
            target_reflectance: (B, 31) - 目标反射率（从训练数据）
            lab_target: (B, 3) - 目标 Lab

        返回:
            total_loss: 标量
            loss_dict: 各项损失的字典（用于监控）
        """
        # 1. 反射率曲线匹配损失（主要）
        reflectance_loss = F.mse_loss(pred_reflectance, target_reflectance)

        # 2. 光谱平滑损失
        smoothness_loss = self.compute_smoothness_loss(pred_reflectance)

        # 3. 物理约束损失（如果需要）
        physics_loss = self.compute_physics_loss(pred_reflectance)

        # 组合损失
        if self.use_auto_balance:
            total_loss = (
                reflectance_loss / (2 * torch.exp(self.log_sigma_color)) +
                self.log_sigma_color +
                smoothness_loss / (2 * torch.exp(self.log_sigma_smooth)) +
                self.log_sigma_smooth
            )
        else:
            total_loss = (
                reflectance_loss +
                self.lambda_smooth * smoothness_loss +
                self.lambda_physics * physics_loss
            )

        # 返回详细损失信息
        loss_dict = {
            'total': total_loss.item(),
            'reflectance': reflectance_loss.item(),
            'smoothness': smoothness_loss.item(),
            'physics': physics_loss.item()
        }

        return total_loss, loss_dict

    def compute_smoothness_loss(self, reflectance):
        """
        计算光谱平滑损失

        惩罚相邻波长点的剧烈变化
        """
        # 一阶导数：|R(λ_{i+1}) - R(λ_i)|^2
        first_order_diff = reflectance[:, 1:] - reflectance[:, :-1]
        smoothness = torch.mean(first_order_diff ** 2)

        return smoothness

    def compute_physics_loss(self, reflectance):
        """
        计算物理约束损失

        确保反射率在 [0, 1] 范围内
        注：如果模型最后用了 Sigmoid，这项基本为 0
        """
        # 惩罚超出边界的值
        lower_violation = F.relu(-reflectance)
        upper_violation = F.relu(reflectance - 1.0)

        physics = torch.mean(lower_violation + upper_violation)

        return physics


class ColorMatchingLoss(nn.Module):
    """
    可选：基于颜色空间的损失（需要可微分的 Lab 计算）
    """

    def __init__(self, illuminant_data):
        super().__init__()
        # 存储光源和观察者函数数据（用于可微分计算）
        self.register_buffer('illuminant_spd', illuminant_data['spd'])
        self.register_buffer('observer_x', illuminant_data['observer_x'])
        self.register_buffer('observer_y', illuminant_data['observer_y'])
        self.register_buffer('observer_z', illuminant_data['observer_z'])

    def reflectance_to_xyz(self, reflectance):
        """
        可微分的反射率 → XYZ 转换

        参数:
            reflectance: (B, 31)

        返回:
            xyz: (B, 3)
        """
        # R(λ) * I(λ) * observer(λ)
        X = torch.sum(
            reflectance * self.illuminant_spd * self.observer_x,
            dim=1
        ) * 10  # 步长 10nm

        Y = torch.sum(
            reflectance * self.illuminant_spd * self.observer_y,
            dim=1
        ) * 10

        Z = torch.sum(
            reflectance * self.illuminant_spd * self.observer_z,
            dim=1
        ) * 10

        # 归一化
        k = 100.0 / torch.sum(self.illuminant_spd * self.observer_y) / 10
        xyz = torch.stack([X * k, Y * k, Z * k], dim=1)

        return xyz

    def xyz_to_lab(self, xyz, illuminant_xyz):
        """
        可微分的 XYZ → Lab 转换

        参数:
            xyz: (B, 3)
            illuminant_xyz: (3,) - 参考白点

        返回:
            lab: (B, 3)
        """
        # 归一化
        xyz_normalized = xyz / illuminant_xyz

        # f(t) 函数
        delta = 6.0 / 29.0
        delta_cubed = delta ** 3

        def f(t):
            # 分段函数
            mask = t > delta_cubed
            result = torch.where(
                mask,
                torch.pow(t, 1.0/3.0),
                t / (3 * delta**2) + 4.0/29.0
            )
            return result

        fx = f(xyz_normalized[:, 0])
        fy = f(xyz_normalized[:, 1])
        fz = f(xyz_normalized[:, 2])

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        lab = torch.stack([L, a, b], dim=1)

        return lab

    def forward(self, pred_reflectance, lab_target):
        """
        计算颜色匹配损失（Delta E）
        """
        # 反射率 → XYZ
        xyz_pred = self.reflectance_to_xyz(pred_reflectance)

        # XYZ → Lab（D65 白点）
        illuminant_xyz = torch.tensor([95.047, 100.0, 108.883])
        lab_pred = self.xyz_to_lab(xyz_pred, illuminant_xyz)

        # 计算 Delta E
        delta_e = torch.sqrt(torch.sum((lab_pred - lab_target) ** 2, dim=1))

        return torch.mean(delta_e)


def print_loss_statistics(loss_history):
    """
    打印训练过程中的损失统计
    """
    print("\n损失统计:")
    print("-" * 60)
    print(f"{'Loss Type':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 60)

    for loss_type in ['total', 'reflectance', 'smoothness', 'physics']:
        values = [epoch[loss_type] for epoch in loss_history]
        print(f"{loss_type:<20} "
              f"{np.mean(values):<12.4f} "
              f"{np.std(values):<12.4f} "
              f"{np.min(values):<12.4f} "
              f"{np.max(values):<12.4f}")

    print("-" * 60)


# 使用示例
if __name__ == '__main__':
    # 创建模型
    model = LabToKSAdapter()
    print(f"模型参数量: {model.count_parameters():,}")

    # 创建损失函数
    criterion = AdapterLoss(
        lambda_smooth=0.1,
        lambda_physics=0.0,  # Sigmoid 已经约束了范围
        use_auto_balance=False
    )

    # 测试前向传播
    batch_size = 8
    lab_input = torch.randn(batch_size, 3)  # 随机 Lab
    target_reflectance = torch.rand(batch_size, 31)  # 随机目标反射率

    # 前向
    pred_reflectance = model(lab_input)
    loss, loss_dict = criterion(pred_reflectance, target_reflectance, lab_input)

    print(f"\n测试输出:")
    print(f"  输入 Lab shape: {lab_input.shape}")
    print(f"  输出反射率 shape: {pred_reflectance.shape}")
    print(f"  反射率范围: [{pred_reflectance.min():.3f}, {pred_reflectance.max():.3f}]")
    print(f"  总损失: {loss.item():.4f}")
    print(f"  损失详情: {loss_dict}")
