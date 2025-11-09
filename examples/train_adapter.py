"""
训练 Lab → K/S 转接层示例

展示如何：
1. 从颜料库生成训练数据
2. 训练转接层
3. 评估性能
4. 调优超参数
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pigment_trace.adapter import LabToKSAdapter, AdapterLoss
from pigment_trace.km.km_core import KubelkaMunkCore
from pigment_trace.km.observer_functions import Observer
from pigment_trace.km.illuminants import D65
from pigment_trace.km.color_converter import ColorConverter


# ============================================================================
# 1. 训练数据生成
# ============================================================================

class PigmentLibrary:
    """模拟颜料库"""

    def __init__(self, n_pigments=20):
        """
        创建模拟颜料库

        实际使用时，应该从真实颜料测量数据加载
        """
        self.n_pigments = n_pigments
        self.wavelengths = np.arange(400, 701, 10)  # 31 个波长点

        # 生成模拟的颜料 K/S 数据
        self.pigments_ks = self._generate_pigments()

    def _generate_pigments(self):
        """生成模拟颜料的 K/S 数据"""
        pigments = []

        for i in range(self.n_pigments):
            # 每个颜料有特定的吸收峰
            peak_wavelength = 400 + i * 300 / self.n_pigments

            # 生成高斯型吸收曲线
            K = np.exp(-((self.wavelengths - peak_wavelength) ** 2) / (2 * 50**2))
            S = np.ones_like(K) * 0.5  # 散射系数

            # 归一化
            K = K / np.max(K) * 2.0

            pigments.append((K, S))

        return pigments

    def mix_pigments(self, concentrations):
        """
        混合颜料

        参数:
            concentrations: (n_pigments,) - 各颜料浓度

        返回:
            K_mixed, S_mixed: 混合后的 K/S 系数
        """
        concentrations = np.array(concentrations)
        concentrations = concentrations / np.sum(concentrations)  # 归一化

        K_mixed = np.zeros(len(self.wavelengths))
        S_mixed = np.zeros(len(self.wavelengths))

        for (K, S), conc in zip(self.pigments_ks, concentrations):
            if conc > 0:
                K_mixed += K * conc
                S_mixed += S * conc

        return K_mixed, S_mixed


def ks_to_reflectance(K, S):
    """K/S → 反射率"""
    return KubelkaMunkCore.reflectance(K, S)


def reflectance_to_lab(reflectance, wavelengths):
    """反射率 → Lab（D65 光源）"""
    observer = Observer()
    illuminant = D65()

    std_wavelengths = observer.get_wavelengths()
    d65_spd = illuminant.get_spd()

    # 插值
    reflectance_interp = np.interp(std_wavelengths, wavelengths, reflectance)

    # 获取观察者函数
    x_bar, y_bar, z_bar = observer.get_xyz_curves()

    # 计算 XYZ
    X = np.sum(reflectance_interp * d65_spd * x_bar) * 10
    Y = np.sum(reflectance_interp * d65_spd * y_bar) * 10
    Z = np.sum(reflectance_interp * d65_spd * z_bar) * 10

    # 归一化
    k = 100.0 / np.sum(d65_spd * y_bar * 10)
    X *= k
    Y *= k
    Z *= k

    # 转换到 Lab
    return ColorConverter.xyz_to_lab(X, Y, Z)


def generate_training_data(pigment_library, n_samples=10000):
    """
    从颜料库生成训练数据

    返回:
        data: List of (lab, reflectance) tuples
    """
    print(f"生成 {n_samples} 个训练样本...")

    data = []
    wavelengths = pigment_library.wavelengths

    for i in range(n_samples):
        # 随机采样配方
        n_pigments = np.random.randint(2, 6)  # 用 2-5 种颜料
        concentrations = np.zeros(pigment_library.n_pigments)

        selected_pigments = np.random.choice(
            pigment_library.n_pigments,
            n_pigments,
            replace=False
        )

        # Dirichlet 分布生成浓度
        selected_concentrations = np.random.dirichlet(np.ones(n_pigments))

        for idx, conc in zip(selected_pigments, selected_concentrations):
            concentrations[idx] = conc

        # 混合 K/S
        K_mixed, S_mixed = pigment_library.mix_pigments(concentrations)

        # 计算反射率
        reflectance = ks_to_reflectance(K_mixed, S_mixed)

        # 计算 Lab
        lab = reflectance_to_lab(reflectance, wavelengths)

        data.append({
            'lab': np.array(lab),
            'reflectance': reflectance,
            'concentrations': concentrations
        })

        if (i + 1) % 1000 == 0:
            print(f"  已生成 {i + 1}/{n_samples} 个样本")

    print("数据生成完成!")
    return data


class AdapterDataset(Dataset):
    """PyTorch Dataset for adapter training"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'lab': torch.FloatTensor(sample['lab']),
            'reflectance': torch.FloatTensor(sample['reflectance'])
        }


# ============================================================================
# 2. 训练函数
# ============================================================================

def train_adapter(
    train_data,
    val_data=None,
    lambda_smooth=0.1,
    lambda_physics=0.0,
    batch_size=256,
    epochs=50,
    lr=1e-3
):
    """
    训练转接层

    参数:
        train_data: 训练数据
        val_data: 验证数据（可选）
        lambda_smooth: 平滑损失权重
        lambda_physics: 物理约束损失权重
        batch_size: 批大小
        epochs: 训练轮数
        lr: 学习率

    返回:
        model: 训练好的模型
        history: 训练历史
    """
    print("\n" + "=" * 60)
    print("开始训练转接层")
    print("=" * 60)
    print(f"训练样本: {len(train_data)}")
    if val_data:
        print(f"验证样本: {len(val_data)}")
    print(f"λ_smooth: {lambda_smooth}")
    print(f"λ_physics: {lambda_physics}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print("=" * 60)

    # 创建模型
    model = LabToKSAdapter()
    print(f"\n模型参数量: {model.count_parameters():,}")

    # 损失函数
    criterion = AdapterLoss(
        lambda_smooth=lambda_smooth,
        lambda_physics=lambda_physics
    )

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DataLoader
    train_dataset = AdapterDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    if val_data:
        val_dataset = AdapterDataset(val_data)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # 训练循环
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_losses = []

        for batch in train_loader:
            lab = batch['lab']
            reflectance_target = batch['reflectance']

            # 前向
            reflectance_pred = model(lab)
            loss, loss_dict = criterion(
                reflectance_pred,
                reflectance_target,
                lab
            )

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 平均训练损失
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # 验证
        if val_data:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    lab = batch['lab']
                    reflectance_target = batch['reflectance']

                    reflectance_pred = model(lab)
                    loss, _ = criterion(
                        reflectance_pred,
                        reflectance_target,
                        lab
                    )

                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss = {avg_train_loss:.4f}")

    print("\n训练完成!")
    return model, history


# ============================================================================
# 3. 超参数调优
# ============================================================================

def tune_hyperparameters(train_data, val_data):
    """
    网格搜索最优超参数
    """
    print("\n" + "=" * 60)
    print("超参数调优")
    print("=" * 60)

    lambda_smooth_grid = [0.01, 0.05, 0.1, 0.2]

    results = []

    for lambda_s in lambda_smooth_grid:
        print(f"\n测试 λ_smooth = {lambda_s}")

        model, history = train_adapter(
            train_data,
            val_data,
            lambda_smooth=lambda_s,
            lambda_physics=0.0,
            epochs=20,  # 少量 epoch 快速测试
            batch_size=256
        )

        final_val_loss = history['val_loss'][-1]

        results.append({
            'lambda_smooth': lambda_s,
            'val_loss': final_val_loss
        })

        print(f"  最终验证损失: {final_val_loss:.4f}")

    # 找最优
    best_result = min(results, key=lambda x: x['val_loss'])

    print("\n" + "=" * 60)
    print("调优结果:")
    print("=" * 60)
    for r in results:
        marker = " ← 最优" if r == best_result else ""
        print(f"  λ_smooth = {r['lambda_smooth']:.3f}, "
              f"Val Loss = {r['val_loss']:.4f}{marker}")

    print(f"\n推荐参数: λ_smooth = {best_result['lambda_smooth']}")

    return best_result


# ============================================================================
# 4. 主程序
# ============================================================================

def main():
    print("=" * 60)
    print("Lab → K/S 转接层训练示例")
    print("=" * 60)

    # 1. 创建模拟颜料库
    print("\n1. 创建颜料库...")
    pigment_library = PigmentLibrary(n_pigments=20)
    print(f"   颜料数量: {pigment_library.n_pigments}")

    # 2. 生成训练数据
    print("\n2. 生成训练数据...")
    all_data = generate_training_data(pigment_library, n_samples=10000)

    # 划分训练/验证集
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    print(f"   训练集: {len(train_data)} 样本")
    print(f"   验证集: {len(val_data)} 样本")

    # 3. 训练模型
    print("\n3. 训练模型...")
    model, history = train_adapter(
        train_data,
        val_data,
        lambda_smooth=0.1,
        lambda_physics=0.0,
        batch_size=256,
        epochs=50,
        lr=1e-3
    )

    # 4. 保存模型
    print("\n4. 保存模型...")
    torch.save(model.state_dict(), 'adapter_model.pth')
    print("   模型已保存到: adapter_model.pth")

    # 5. 测试推理
    print("\n5. 测试推理...")
    model.eval()

    test_lab = torch.FloatTensor([[50.0, 0.0, 0.0]])  # 中性灰
    with torch.no_grad():
        pred_reflectance = model(test_lab)

    print(f"   输入 Lab: {test_lab[0].numpy()}")
    print(f"   输出反射率: shape={pred_reflectance.shape}, "
          f"范围=[{pred_reflectance.min():.3f}, {pred_reflectance.max():.3f}]")
    print(f"   平滑度: {torch.mean((pred_reflectance[:, 1:] - pred_reflectance[:, :-1])**2).item():.6f}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
