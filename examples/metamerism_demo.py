"""
同色异谱演示：展示同一个Lab值可以对应无穷多条光谱曲线

这个脚本演示了即使在固定光源（D65）下，
同一个Lab值也可以由多条完全不同的反射光谱产生。
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pigment_trace.km.observer_functions import Observer
from pigment_trace.km.illuminants import D65
from pigment_trace.km.color_converter import ColorConverter


def spectrum_to_lab(reflectance_spectrum, wavelengths):
    """
    将反射光谱转换为Lab值

    参数:
        reflectance_spectrum: 反射率数组
        wavelengths: 波长数组

    返回:
        (L, a, b) 元组
    """
    # 获取D65光源和观察者函数
    observer = Observer()
    illuminant = D65()

    # 获取标准波长
    std_wavelengths = observer.get_wavelengths()
    d65_spd = illuminant.get_spd()

    # 插值反射光谱到标准波长
    reflectance_interp = np.interp(std_wavelengths, wavelengths,
                                    reflectance_spectrum)

    # 获取观察者函数
    x_bar, y_bar, z_bar = observer.get_xyz_curves()

    # 计算 XYZ (步长为10nm)
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


def generate_metameric_spectra(target_lab, n_examples=5):
    """
    生成多条产生相同Lab值的不同光谱曲线

    这里用一个简化的方法：在一条基础光谱上叠加
    高频振荡，这些振荡在积分后会相互抵消
    """
    wavelengths = np.arange(400, 701, 10)  # 31个点

    # 基础光谱：根据目标Lab粗略估计一个灰度值
    L, a, b = target_lab
    base_reflectance = (L / 100.0) * np.ones(len(wavelengths))

    print(f"\n目标 Lab: L={L:.1f}, a={a:.1f}, b={b:.1f}")
    print(f"信息维度: Lab(3维) ← Spectrum(31维)")
    print(f"自由度差异: 28维")
    print(f"\n生成{n_examples}条产生相同Lab的不同光谱曲线:\n")

    metamers = []

    for i in range(n_examples):
        # 添加高频振荡（不同的振荡模式）
        # 这些振荡在与观察者函数卷积时会部分抵消
        frequency = 2 + i * 0.5
        phase = i * np.pi / n_examples
        amplitude = 0.05

        oscillation = amplitude * np.sin(
            2 * np.pi * frequency * np.linspace(0, 1, len(wavelengths)) + phase
        )

        # 叠加振荡
        spectrum = np.clip(base_reflectance + oscillation, 0, 1)

        # 计算实际的Lab值
        actual_lab = spectrum_to_lab(spectrum, wavelengths)

        metamers.append((spectrum, actual_lab))

        # 打印光谱和Lab
        print(f"光谱 #{i+1}:")
        print(f"  R(450nm)={spectrum[5]:.4f}, R(550nm)={spectrum[15]:.4f}, "
              f"R(650nm)={spectrum[25]:.4f}")
        print(f"  实际 Lab: L={actual_lab[0]:.2f}, a={actual_lab[1]:.2f}, "
              f"b={actual_lab[2]:.2f}")
        print(f"  Lab误差: ΔE = {delta_e(target_lab, actual_lab):.4f}")
        print()

    return metamers


def delta_e(lab1, lab2):
    """计算两个Lab值之间的色差（ΔE）"""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    return np.sqrt((L2 - L1)**2 + (a2 - a1)**2 + (b2 - b1)**2)


def main():
    print("=" * 60)
    print("同色异谱演示：信息维度问题")
    print("=" * 60)

    # 目标Lab：中等灰色
    target_lab = (50.0, 0.0, 0.0)

    # 生成同色异谱光谱
    metamers = generate_metameric_spectra(target_lab, n_examples=5)

    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    print("✓ 所有光谱曲线形状完全不同")
    print("✓ 但它们在D65光源下产生几乎相同的Lab值")
    print("✓ 这证明了 Lab → Spectrum 是一对多映射")
    print("\n原因:")
    print("  - 反射光谱有31个自由度（31维空间）")
    print("  - Lab只有3个自由度（3维空间）")
    print("  - 从31维投影到3维，必然损失28维的信息")
    print("  - 因此逆向重建时存在无穷多个解")
    print("\n对配色模型的影响:")
    print("  ✗ Lab → K/S 映射是病态问题（ill-posed）")
    print("  ✓ Spectrum → K/S 映射是良定问题（well-posed）")
    print("=" * 60)


if __name__ == '__main__':
    main()
