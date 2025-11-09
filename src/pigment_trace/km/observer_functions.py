"""
CIE 标准观察者函数

数据来源: https://cie.co.at/data-tables
波长范围: 380-750nm, 10nm 步长
"""

import numpy as np

# 波长参数
SPD_MIN_NM = 380
SPD_MAX_NM = 750
SPD_STEP_SIZE_NM = 10
SPD_BUCKETS = (SPD_MAX_NM - SPD_MIN_NM) // SPD_STEP_SIZE_NM + 1  # 38


class Observer:
    """CIE 标准观察者"""

    def __init__(self):
        self.wavelengths = np.arange(SPD_MIN_NM, SPD_MAX_NM + 1, SPD_STEP_SIZE_NM)

        # CIE 1931 2° 标准观察者函数
        self.X = np.array([
            0.0002, 0.0024, 0.0191, 0.0847, 0.2045, 0.3147, 0.3837, 0.3707,
            0.3023, 0.1956, 0.0805, 0.0162, 0.0038, 0.0375, 0.1177, 0.2365,
            0.3768, 0.5298, 0.7052, 0.8787, 1.0142, 1.1185, 1.124, 1.0305,
            0.8563, 0.6475, 0.4316, 0.2683, 0.1526, 0.0813, 0.0409, 0.0199,
            0.0096, 0.0046, 0.0022, 0.001, 0.0005, 0.0003
        ])

        self.Y = np.array([
            0, 0.0003, 0.002, 0.0088, 0.0214, 0.0387, 0.0621, 0.0895, 0.1282,
            0.1852, 0.2536, 0.3391, 0.4608, 0.6067, 0.7618, 0.8752, 0.962,
            0.9918, 0.9973, 0.9556, 0.8689, 0.7774, 0.6583, 0.528, 0.3981,
            0.2835, 0.1798, 0.1076, 0.0603, 0.0318, 0.0159, 0.0077, 0.0037,
            0.0018, 0.0008, 0.0004, 0.0002, 0.0001
        ])

        self.Z = np.array([
            0.0007, 0.0105, 0.086, 0.3894, 0.9725, 1.5535, 1.9673, 1.9948,
            1.7454, 1.3176, 0.7721, 0.4153, 0.2185, 0.112, 0.0607, 0.0305,
            0.0137, 0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0
        ])

        assert len(self.X) == SPD_BUCKETS
        assert len(self.Y) == SPD_BUCKETS
        assert len(self.Z) == SPD_BUCKETS

    def get_xyz_curves(self):
        """获取 XYZ 曲线"""
        return self.X, self.Y, self.Z

    def get_wavelengths(self):
        """获取波长数组"""
        return self.wavelengths
