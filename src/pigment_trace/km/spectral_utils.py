"""
光谱数据处理工具
"""

import numpy as np
from scipy import interpolate


def interpolate_spectrum(wavelengths_in, values_in, wavelengths_out,
                         method='cubic'):
    """
    插值光谱数据到目标波长
    
    参数:
        wavelengths_in: 输入波长数组
        values_in: 输入值数组
        wavelengths_out: 输出波长数组
        method: 插值方法 ('linear', 'cubic', 'quadratic')
        
    返回:
        插值后的值数组
    """
    # 确保输入是 numpy 数组
    wavelengths_in = np.array(wavelengths_in)
    values_in = np.array(values_in)
    wavelengths_out = np.array(wavelengths_out)

    # 检查输入
    if len(wavelengths_in) != len(values_in):
        raise ValueError("波长和值的数量必须相同")

    # 排序（如果未排序）
    sort_idx = np.argsort(wavelengths_in)
    wavelengths_in = wavelengths_in[sort_idx]
    values_in = values_in[sort_idx]

    # 创建插值函数
    if method == 'linear':
        f = interpolate.interp1d(wavelengths_in, values_in,
                                 kind='linear', bounds_error=False,
                                 fill_value=(values_in[0], values_in[-1]))
    elif method == 'cubic':
        # 如果数据点太少，降级到线性
        if len(wavelengths_in) < 4:
            f = interpolate.interp1d(wavelengths_in, values_in,
                                     kind='linear', bounds_error=False,
                                     fill_value=(values_in[0], values_in[-1]))
        else:
            f = interpolate.interp1d(wavelengths_in, values_in,
                                     kind='cubic', bounds_error=False,
                                     fill_value=(values_in[0], values_in[-1]))
    elif method == 'quadratic':
        if len(wavelengths_in) < 3:
            f = interpolate.interp1d(wavelengths_in, values_in,
                                     kind='linear', bounds_error=False,
                                     fill_value=(values_in[0], values_in[-1]))
        else:
            f = interpolate.interp1d(wavelengths_in, values_in,
                                     kind='quadratic', bounds_error=False,
                                     fill_value=(values_in[0], values_in[-1]))
    else:
        raise ValueError(f"不支持的插值方法: {method}")

    # 执行插值
    return f(wavelengths_out)


def resample_ks_data(k_data, s_data, wavelengths_in, wavelengths_out,
                     method='cubic'):
    """
    重采样 K/S 数据到目标波长
    
    参数:
        k_data: K 系数数组
        s_data: S 系数数组
        wavelengths_in: 输入波长数组
        wavelengths_out: 输出波长数组
        method: 插值方法
        
    返回:
        (K_resampled, S_resampled) 元组
    """
    K_resampled = interpolate_spectrum(wavelengths_in, k_data,
                                       wavelengths_out, method)
    S_resampled = interpolate_spectrum(wavelengths_in, s_data,
                                       wavelengths_out, method)

    # 确保 K 和 S 非负
    K_resampled = np.maximum(K_resampled, 0)
    S_resampled = np.maximum(S_resampled, 0)

    return K_resampled, S_resampled
