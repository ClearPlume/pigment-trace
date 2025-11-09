"""
颜色空间转换
"""

import numpy as np


class ColorConverter:
    """颜色空间转换器"""

    # XYZ 到 RGB 转换矩阵 (sRGB, D65)
    XYZ_TO_RGB = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])

    # RGB 到 XYZ 转换矩阵
    RGB_TO_XYZ = np.linalg.inv(XYZ_TO_RGB)

    @staticmethod
    def xyz_to_rgb(X, Y, Z):
        """XYZ 转 RGB"""
        xyz = np.array([X, Y, Z])
        rgb = ColorConverter.XYZ_TO_RGB @ xyz
        return rgb

    @staticmethod
    def rgb_to_srgb(rgb):
        """线性 RGB 转 sRGB (gamma 校正)"""
        return np.where(rgb <= 0.0031308,
                        12.92 * rgb,
                        1.055 * np.power(rgb, 1 / 2.4) - 0.055)

    @staticmethod
    def srgb_to_rgb(srgb):
        """sRGB 转线性 RGB"""
        return np.where(srgb <= 0.04045,
                        srgb / 12.92,
                        np.power((srgb + 0.055) / 1.055, 2.4))

    @staticmethod
    def xyz_to_lab(X, Y, Z, illuminant_xyz=(95.047, 100.0, 108.883)):
        """
        XYZ 转 Lab
        
        参数:
            X, Y, Z: XYZ 值
            illuminant_xyz: 参考白点的 XYZ 值 (默认 D65)
        """
        Xn, Yn, Zn = illuminant_xyz

        def f(t):
            delta = 6 / 29
            return np.where(t > delta ** 3,
                            np.power(t, 1 / 3),
                            t / (3 * delta ** 2) + 4 / 29)

        fx = f(X / Xn)
        fy = f(Y / Yn)
        fz = f(Z / Zn)

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return L, a, b

    @staticmethod
    def lab_to_xyz(L, a, b, illuminant_xyz=(95.047, 100.0, 108.883)):
        """
        Lab 转 XYZ
        
        参数:
            L, a, b: Lab 值
            illuminant_xyz: 参考白点的 XYZ 值 (默认 D65)
        """
        Xn, Yn, Zn = illuminant_xyz

        def f_inv(t):
            delta = 6 / 29
            return np.where(t > delta,
                            np.power(t, 3),
                            3 * delta ** 2 * (t - 4 / 29))

        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200

        X = Xn * f_inv(fx)
        Y = Yn * f_inv(fy)
        Z = Zn * f_inv(fz)

        return X, Y, Z
