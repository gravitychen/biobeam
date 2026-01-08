#!/usr/bin/env python
# coding: utf-8

# # PSF 截面比较可视化
# 
# 本脚本用于加载和比较不同 PSF 的 XY 和 XZ 截面

import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import os

# 设置字体为 Liberation
plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'Liberation Serif', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# PSF 文件路径
psf_files = {
    'Cylindrical Lightsheet': 'cylindrical_lightsheet_effective_psf.tif',
    'Bessel Lattice Lightsheet': 'bessel_lattice_lightsheet_effective_psf.tif',
    'Confocal': 'confocal_psf.tif',
    'Widefield': 'widefield_effective.tif'
}

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

print("正在加载 PSF 数据...")
psf_data = {}
for name, filename in psf_files.items():
    filepath = os.path.join(script_dir, filename)
    if os.path.exists(filepath):
        print(f"加载 {name}: {filepath}")
        psf = imread(filepath)
        print(f"  {name} 形状: {psf.shape}")
        print(f"  {name} 最大值: {psf.max():.4f}")
        psf_normalized = psf / psf.max()
        psf_data[name] = psf_normalized
    else:
        print(f"警告: 文件不存在 {filepath}")

if not psf_data:
    print("错误: 没有找到任何 PSF 文件！")
    exit(1)

# 提取中心切片
print("\n正在提取中心切片...")
slices_xy = {}
slices_xz = {}

for name, psf in psf_data.items():
    # PSF 数据形状为 (Nz, Ny, Nx)，即 (Z, Y, X)
    Nz, Ny, Nx = psf.shape
    center_z = Nz // 2  # Z 方向的中心（光轴中心）
    center_y = Ny // 2  # Y 方向的中心
    
    # XY 平面：固定 Z（光轴），取 (Y, X)
    slice_xy = psf[center_z, :, :]
    # XZ 平面：固定 Y，取 (Z, X)
    slice_xz = psf[:, center_y, :]
    
    # 先应用 log1p 变换，然后 Min-Max 归一化到 0-1 范围
    def min_max_norm(data):
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        else:
            return data
    
    # 先 log1p，再归一化
    slice_xy_log = np.log1p(slice_xy)
    slice_xz_log = np.log1p(slice_xz)
    slices_xy[name] = min_max_norm(slice_xy_log)
    slices_xz[name] = min_max_norm(slice_xz_log)
    print(f"{name}:")
    print(f"  XY 截面形状: {slice_xy.shape}")
    print(f"  XZ 截面形状: {slice_xz.shape}")

# 创建比较可视化
print("\n正在生成比较可视化...")

n_psfs = len(psf_data)
fig, axes = plt.subplots(2, n_psfs, figsize=(5*n_psfs, 10))

# 设置颜色范围（已归一化到 0-1，直接使用 1.0）
vmax_xy_norm = 1.0
vmax_xz_norm = 1.0

for idx, (name, psf) in enumerate(psf_data.items()):
    # XY 截面（第一行）
    ax_xy = axes[0, idx]
    im_xy = ax_xy.imshow(slices_xy[name], cmap='turbo', vmax=vmax_xy_norm, aspect='auto', origin='lower')
    ax_xy.set_title(f'{name} PSF\nXY Slice', fontsize=12, fontweight='bold')
    ax_xy.set_xlabel('X (pixels)')
    ax_xy.set_ylabel('Y (pixels)')
    plt.colorbar(im_xy, ax=ax_xy, fraction=0.046, pad=0.04, label='Normalized Intensity (0-1)')
    
    # XZ 截面（第二行）
    ax_xz = axes[1, idx]
    im_xz = ax_xz.imshow(slices_xz[name], cmap='turbo', vmax=vmax_xz_norm, aspect='auto', origin='lower')
    ax_xz.set_title(f'XZ Slice', fontsize=12, fontweight='bold')
    ax_xz.set_xlabel('X (pixels)')
    ax_xz.set_ylabel('Z (pixels)')
    plt.colorbar(im_xz, ax=ax_xz, fraction=0.046, pad=0.04, label='Normalized Intensity (0-1)')

plt.tight_layout()
# plt.suptitle('PSF Slice Comparison - Detection Objective Coordinate System (Log1p + Min-Max Normalized 0-1)\nZ is optical axis, XY is lateral plane', 
#              fontsize=14, y=0.995)

# 保存图像
output_file = os.path.join(script_dir, 'psf_slices_comparison.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n比较图像已保存到: {output_file}")

plt.show()
print("可视化完成！")

