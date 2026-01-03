#!/usr/bin/env python
# coding: utf-8

# # PSF 截面比较可视化
# 
# 本脚本用于加载和比较不同 PSF 的 XY 和 XZ 截面

import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# PSF 文件路径
psf_files = {
    'Cylindrical Lightsheet': 'cylindrical_lightsheet_effective_psf.tif',
    'Bessel Lattice Lightsheet': 'bessel_lattice_lightsheet_effective_psf.tif',
    'Confocal': 'confocal_3d.tif',
    'Widefield': 'widefield_3d.tif'
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
    
    # 应用对数变换：log(psf_data + 1)
    slices_xy[name] = np.log1p(slice_xy)
    slices_xz[name] = np.log1p(slice_xz)
    print(f"{name}:")
    print(f"  XY 截面形状: {slice_xy.shape}")
    print(f"  XZ 截面形状: {slice_xz.shape}")

# 创建比较可视化
print("\n正在生成比较可视化...")

n_psfs = len(psf_data)
fig, axes = plt.subplots(2, n_psfs, figsize=(5*n_psfs, 10))

# 设置颜色范围（使用对数变换后的数据）
vmax_xy = max([s.max() for s in slices_xy.values()])
vmax_xz = max([s.max() for s in slices_xz.values()])

# 使用归一化的颜色范围，便于比较
vmax_xy_norm = vmax_xy * 0.8  # 使用 80% 的最大值，避免极值影响显示
vmax_xz_norm = vmax_xz * 0.8

for idx, (name, psf) in enumerate(psf_data.items()):
    # XY 截面（第一行）
    ax_xy = axes[0, idx]
    im_xy = ax_xy.imshow(slices_xy[name], cmap='hot', vmax=vmax_xy_norm, aspect='auto', origin='lower')
    ax_xy.set_title(f'{name}\nXY 截面 (Z=0, Log Scale)', fontsize=12, fontweight='bold')
    ax_xy.set_xlabel('X (像素)')
    ax_xy.set_ylabel('Y (像素)')
    plt.colorbar(im_xy, ax=ax_xy, fraction=0.046, pad=0.04, label='log(强度+1)')
    
    # XZ 截面（第二行）
    ax_xz = axes[1, idx]
    im_xz = ax_xz.imshow(slices_xz[name], cmap='hot', vmax=vmax_xz_norm, aspect='auto', origin='lower')
    ax_xz.set_title(f'{name}\nXZ 截面 (Y=0, Log Scale)', fontsize=12, fontweight='bold')
    ax_xz.set_xlabel('X (像素)')
    ax_xz.set_ylabel('Z (像素)')
    plt.colorbar(im_xz, ax=ax_xz, fraction=0.046, pad=0.04, label='log(强度+1)')

plt.tight_layout()
plt.suptitle('PSF 截面比较 - Detection Objective 坐标系 (对数尺度)\nZ 是光轴，XY 是横向平面', 
             fontsize=14, y=0.995)

# 保存图像
output_file = os.path.join(script_dir, 'psf_slices_comparison.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n比较图像已保存到: {output_file}")

plt.show()
print("可视化完成！")

