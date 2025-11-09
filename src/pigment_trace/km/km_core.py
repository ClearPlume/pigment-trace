"""
Kubelka-Munk 理论核心算法

MIT License
"""

import numpy as np


class KubelkaMunkCore:
    """Kubelka-Munk 理论核心实现"""

    # Saunderson 校正常数
    K1 = 0.0031
    K2 = 0.650

    @staticmethod
    def reflectance(K, S):
        """
        计算反射率 R
        
        参数:
            K: 吸收系数或吸收系数数组
            S: 散射系数或散射系数数组
            
        返回:
            反射率
        """
        # 处理 S=0 的情况
        S = np.where(S == 0, 1e-10, S)

        ks = K / S
        # R = 1 + K/S - sqrt((K/S)^2 + 2*K/S)
        return 1.0 + ks - np.sqrt(ks * ks + 2.0 * ks)

    @classmethod
    def reflectance_with_saunderson(cls, K, S):
        """
        应用 Saunderson 校正的反射率计算
        
        参数:
            K: 吸收系数
            S: 散射系数
            
        返回:
            校正后的反射率
        """
        R = cls.reflectance(K, S)
        # Saunderson 校正公式
        return ((1.0 - cls.K1) * (1.0 - cls.K2) * R) / (1.0 - cls.K2 * R)

    @staticmethod
    def mix_pigments(pigments_ks, concentrations):
        """
        混合多个颜料的 K 和 S 值
        
        参数:
            pigments_ks: 颜料 K/S 系数列表 [(K_array, S_array), ...]
            concentrations: 浓度数组，对应每个颜料的浓度
            
        返回:
            混合后的 (K, S) 元组
        """
        if len(pigments_ks) != len(concentrations):
            raise ValueError("颜料数量必须与浓度数量相同")

        # 初始化
        K_mixed = np.zeros_like(pigments_ks[0][0], dtype=float)
        S_mixed = np.zeros_like(pigments_ks[0][1], dtype=float)

        # 线性混合 (加法混合)
        for (K, S), concentration in zip(pigments_ks, concentrations):
            K_mixed += np.array(K) * concentration
            S_mixed += np.array(S) * concentration

        return K_mixed, S_mixed

    @classmethod
    def compute_reflectance_spectrum(cls, pigments_ks, concentrations,
                                     use_saunderson=True):
        """
        计算混合颜料的反射光谱
        
        参数:
            pigments_ks: 颜料 K/S 系数列表
            concentrations: 浓度列表
            use_saunderson: 是否应用 Saunderson 校正
            
        返回:
            反射率光谱 (numpy array)
        """
        K_mixed, S_mixed = cls.mix_pigments(pigments_ks, concentrations)

        if use_saunderson:
            return cls.reflectance_with_saunderson(K_mixed, S_mixed)
        else:
            return cls.reflectance(K_mixed, S_mixed)
