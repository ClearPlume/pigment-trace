"""
CIE 标准光源

数据来源: https://cie.co.at/data-tables
波长范围: 380-750nm, 10nm 步长
"""

import numpy as np

from .observer_functions import SPD_BUCKETS


class Illuminant:
    """标准光源基类"""

    def __init__(self, spd):
        """
        参数:
            spd: 光谱功率分布 (Spectral Power Distribution)
        """
        if len(spd) != SPD_BUCKETS:
            raise ValueError(f"SPD 必须有 {SPD_BUCKETS} 个数据点")
        self.spd = np.array(spd)

    def get_spd(self):
        """获取光谱功率分布"""
        return self.spd


class D65(Illuminant):
    """CIE 标准光源 D65 (日光, 6500K)"""

    def __init__(self):
        # D65 光谱功率分布
        spd = [
            49.9755, 54.6482, 82.7549, 91.486, 93.4318, 86.6823, 104.865,
            117.008, 117.812, 114.861, 115.923, 108.811, 109.354, 107.802,
            104.79, 107.689, 104.405, 104.046, 100, 96.3342, 95.788, 88.6856,
            90.0062, 89.5991, 87.6987, 83.2886, 83.6992, 80.0268, 80.2146,
            82.2778, 78.2842, 69.7213, 71.6091, 74.349, 61.604, 69.8856,
            75.087, 63.5927
        ]
        super().__init__(spd)


class A(Illuminant):
    """CIE 标准光源 A (白炽灯, 2856K)"""

    def __init__(self):
        # 简化的 A 光源数据（实际应用中应使用完整数据）
        spd = [
            9.8, 10.9, 12.1, 13.4, 14.7, 16.2, 17.8, 19.5, 21.3, 23.2,
            25.1, 27.1, 29.3, 31.6, 34.0, 36.5, 39.1, 41.8, 44.6, 47.4,
            50.4, 53.5, 56.6, 59.9, 63.3, 66.8, 70.4, 74.1, 77.9, 81.9,
            85.9, 90.0, 94.2, 98.5, 102.8, 107.2, 111.7, 116.2
        ]
        super().__init__(spd)


class E(Illuminant):
    """CIE 标准光源 E (等能光源)"""

    def __init__(self):
        # 等能光源：所有波长功率相同
        spd = [100.0] * SPD_BUCKETS
        super().__init__(spd)
