"""
基于 Kubelka-Munk 理论的颜料混合工具
"""

import numpy as np

from color_converter import ColorConverter
from illuminants import D65
from km_core import KubelkaMunkCore
from observer_functions import Observer
from spectral_utils import resample_ks_data


class ColorMixer:
    """颜料混合器"""

    def __init__(self, illuminant=None, observer=None):
        """
        参数:
            illuminant: 光源对象，默认 D65
            observer: 观察者对象，默认 CIE 1931 2°
        """
        self.km = KubelkaMunkCore()
        self.illuminant = illuminant or D65()
        self.observer = observer or Observer()
        self.converter = ColorConverter()

        # 获取观察者函数和波长
        self.X_bar, self.Y_bar, self.Z_bar = self.observer.get_xyz_curves()
        self.wavelengths = self.observer.get_wavelengths()

        # 计算 D65 * XYZ 用于后续计算
        illum_spd = self.illuminant.get_spd()
        self.D65X = illum_spd * self.X_bar
        self.D65Y = illum_spd * self.Y_bar
        self.D65Z = illum_spd * self.Z_bar

        # 计算归一化因子 (Y 的积分)
        self.Y_norm = np.trapz(self.D65Y, self.wavelengths)

    def mix(self, pigments, concentrations, use_saunderson=True):
        """
        混合颜料并返回 Lab 值
        
        参数:
            pigments: 颜料列表，每个颜料是包含 'k' 和 's' 的字典
                     或 (K_array, S_array) 元组
            concentrations: 浓度列表
            use_saunderson: 是否使用 Saunderson 校正
            
        返回:
            (L, a, b) 元组
        """
        # 规范化输入格式
        pigments_ks = []
        for pig in pigments:
            if isinstance(pig, dict):
                K, S = self._prepare_pigment_data(pig)
                pigments_ks.append((K, S))
            elif isinstance(pig, (tuple, list)) and len(pig) == 2:
                K, S = self._prepare_pigment_data_from_arrays(pig[0], pig[1])
                pigments_ks.append((K, S))
            else:
                raise ValueError("颜料格式错误")

        # 计算混合后的反射光谱
        reflectance = self.km.compute_reflectance_spectrum(
            pigments_ks, concentrations, use_saunderson
        )

        # 反射光谱转 XYZ
        X, Y, Z = self._reflectance_to_xyz(reflectance)

        # XYZ 转 Lab
        L, a, b = self.converter.xyz_to_lab(X, Y, Z)

        return L, a, b

    def _prepare_pigment_data(self, pigment_dict):
        """准备颜料数据（处理插值）"""
        if 'wavelengths' in pigment_dict:
            # 需要插值
            K, S = resample_ks_data(
                pigment_dict['k'],
                pigment_dict['s'],
                pigment_dict['wavelengths'],
                self.wavelengths
            )
        else:
            # 假设已经是正确的波长
            K = np.array(pigment_dict['k'])
            S = np.array(pigment_dict['s'])

            if len(K) != len(self.wavelengths):
                raise ValueError(
                    f"K/S 数据长度 ({len(K)}) 与波长数 ({len(self.wavelengths)}) 不匹配"
                )

        return K, S

    def _prepare_pigment_data_from_arrays(self, K, S):
        """从数组准备颜料数据"""
        K = np.array(K)
        S = np.array(S)

        if len(K) != len(self.wavelengths):
            raise ValueError(
                f"K/S 数据长度 ({len(K)}) 与波长数 ({len(self.wavelengths)}) 不匹配"
            )

        return K, S

    def _reflectance_to_xyz(self, reflectance):
        """反射光谱转 XYZ"""
        # 计算积分 (梯形法则)
        X = np.trapz(reflectance * self.D65X, self.wavelengths)
        Y = np.trapz(reflectance * self.D65Y, self.wavelengths)
        Z = np.trapz(reflectance * self.D65Z, self.wavelengths)

        # 归一化
        X = X / self.Y_norm * 100
        Y = Y / self.Y_norm * 100
        Z = Z / self.Y_norm * 100

        return X, Y, Z

    def get_reflectance_spectrum(self, pigments, concentrations,
                                 use_saunderson=True):
        """
        获取混合后的反射光谱
        
        返回:
            (wavelengths, reflectance) 元组
        """
        pigments_ks = []
        for pig in pigments:
            if isinstance(pig, dict):
                K, S = self._prepare_pigment_data(pig)
                pigments_ks.append((K, S))
            elif isinstance(pig, (tuple, list)) and len(pig) == 2:
                K, S = self._prepare_pigment_data_from_arrays(pig[0], pig[1])
                pigments_ks.append((K, S))
            else:
                raise ValueError("颜料格式错误")

        reflectance = self.km.compute_reflectance_spectrum(
            pigments_ks, concentrations, use_saunderson
        )

        return self.wavelengths.copy(), reflectance
