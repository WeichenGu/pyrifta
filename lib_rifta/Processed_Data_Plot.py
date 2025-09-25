# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 03:46:46 2025

@author: Etrr
"""

import numpy as np
import matplotlib.pyplot as plt

def create_subplot(ax, X, Y, Z, fraction_bar, cbar_label, title, cmap='viridis'):
    """
    在指定的 Axes 上绘制 pcolormesh 图，并添加颜色条和标题
    """
    mesh = ax.pcolormesh(X, Y, Z, cmap=cmap)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title)
    aspect_val = np.shape(Z)[0] / np.shape(Z)[1] * ((1 - fraction_bar) / fraction_bar)
    cbar = plt.colorbar(mesh, ax=ax, fraction=fraction_bar, pad=0.05, shrink=1, aspect=aspect_val)
    cbar.set_label(cbar_label)
    return mesh

# ------------------------------
# 封装绘制一行 4 个子图的扩展方法面板（高度、驻留时间、残差、去除效果）
def plot_extension_row(fig, grid, row, X_data, Y_data, Z_data, T_data, residual_data, removal_data, label_prefix, fraction_bar):
    # Panel 1：高度图（单位转换：m->mm, m->nm）
    ax1 = fig.add_subplot(grid[row, 0])
    title1 = f"{label_prefix}: \nPV = {np.round((np.max(Z_data) - np.min(Z_data)) * 1e9, 2)} nm, " \
             f"RMS = {np.round(np.std(Z_data) * 1e9, 2)} nm"
    create_subplot(ax1, X_data * 1e3, Y_data * 1e3, Z_data * 1e9, fraction_bar, '[nm]', title1)
    
    # Panel 2：驻留时间图（单位：s）
    ax2 = fig.add_subplot(grid[row, 1])
    title2 = f"{label_prefix}: \ndwell time = {np.round(np.sum(T_data), 2)} s"
    create_subplot(ax2, X_data * 1e3, Y_data * 1e3, T_data, fraction_bar, '[s]', title2)
    
    # Panel 3：残差图（单位：nm）
    ax3 = fig.add_subplot(grid[row, 2])
    title3 = f"{label_prefix}: residual \nPV = {np.round((np.max(residual_data) - np.min(residual_data)) * 1e9, 2)} nm, " \
             f"RMS = {np.round(np.std(residual_data) * 1e9, 2)} nm"
    create_subplot(ax3, X_data * 1e3, Y_data * 1e3, residual_data * 1e9, fraction_bar, '[nm]', title3)
    
    # Panel 4：去除效果图（单位：nm）
    ax4 = fig.add_subplot(grid[row, 3])
    title4 = f"{label_prefix}: removal \nPV = {np.round((np.max(removal_data) - np.min(removal_data)) * 1e9, 2)} nm, " \
             f"RMS = {np.round(np.std(removal_data) * 1e9, 2)} nm"
    create_subplot(ax4, X_data * 1e3, Y_data * 1e3, removal_data * 1e9, fraction_bar, '[nm]', title4)
    
    # 返回各子图的颜色条范围（用于统一设置）
    clim_values = (
        ax1.collections[0].get_clim(),
        ax2.collections[0].get_clim(),
        ax3.collections[0].get_clim(),
        ax4.collections[0].get_clim()
    )
    return clim_values

# ------------------------------
# 将每个扩展方法的绘制及保存封装成一个函数
def process_extension_section(fig, grid, plot_idx, selection_flag, save_flag, label, 
                              X_data, Y_data, Z_data, T_data, residual_data, removal_data,
                              map_name, testname, brf_params, fraction_bar,
                              folderjson="./json", m_per_pixel=1, selected_region_length=100, selected_region_width=100,
                              axes_dict=None):
    """
    若 selection_flag 为 True，则绘制一行面板，并保存数据（若 save_flag 为 True）。
    如果 axes_dict 非 None，则把本 section 的颜色范围保存到 axes_dict['zlim']（适用于 Zero extension）。
    返回更新后的 plot_idx。
    """
    if selection_flag:
        clim = plot_extension_row(fig, grid, plot_idx, X_data, Y_data, Z_data, T_data, residual_data, removal_data, label, fraction_bar)
        # 若有需要保存 Zero extension 的颜色范围
        if axes_dict is not None and label.lower().startswith("zero"):
            axes_dict['zlim'] = clim
        plot_idx += 1
        if save_flag:
            simulated_result = {
                map_name[0]: Z_data * 1e9,
                map_name[1]: T_data,
                map_name[2]: residual_data * 1e9,
                map_name[3]: removal_data * 1e9
            }
            # 此处调用保存函数，示例中仅打印信息，实际请调用 Scanfile_Savejson 等函数
            # print(f"保存数据：{testname}_{label.replace(' ', '_')}")
            # for key in map_name:
            #     print(f"  {key}: shape {simulated_result[key].shape}")
    return plot_idx

