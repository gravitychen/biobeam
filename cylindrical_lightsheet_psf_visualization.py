#!/usr/bin/env python
# coding: utf-8

# # Cylindrical Lens Lightsheet PSF 可视化
# 
# 本 notebook 用于可视化 cylindrical lens lightsheet 的：
# 1. **Excitation PSF** - 激发点扩散函数（使用圆柱透镜）
# 2. **Detection PSF** - 检测点扩散函数（使用普通光束）
# 3. **Effective PSF** - 有效点扩散函数（excitation 和 detection PSF 的乘积）
# 

# In[31]:


# Python 3.12+ 兼容性补丁：修复 gputools 中的 SafeConfigParser 导入错误
# 在 Python 3.12+ 中，SafeConfigParser 已被移除，应该使用 ConfigParser
import sys
import configparser

# 检查 Python 版本并应用补丁
if sys.version_info >= (3, 12):
    # 为 gputools 提供 SafeConfigParser 的别名
    if not hasattr(configparser, 'SafeConfigParser'):
        configparser.SafeConfigParser = configparser.ConfigParser
    print("已应用 Python 3.12+ 兼容性补丁：SafeConfigParser -> ConfigParser")
else:
    print(f"Python 版本: {sys.version_info.major}.{sys.version_info.minor}，无需补丁")


# In[32]:


import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.ndimage import rotate
from biobeam.core.focus_field_cylindrical import focus_field_cylindrical
from biobeam.core.focus_field_beam import focus_field_beam

# 设置参数
shape = (128, 128, 128)  # (Nx, Ny, Nz) - 注意：cylindrical lens 在 yz 平面
shape = (52, 52, 52)  # (Nx, Ny, Nz) - 注意：cylindrical lens 在 yz 平面
units = (0.1, 0.1, 0.1)  # (dx, dy, dz) 单位：微米

# 激发参数
lam_excitation = 0.488  # 激发波长（微米）
NA_excitation = 0.2      # 激发数值孔径
n0 = 1.33                # 介质折射率

# 检测参数
lam_detection = 0.525    # 检测波长（微米）
NA_detection = 0.9       # 检测数值孔径

print("参数设置完成")
print(f"形状: {shape}")
print(f"像素大小: {units} 微米")
print(f"激发: λ={lam_excitation} μm, NA={NA_excitation}")
print(f"检测: λ={lam_detection} μm, NA={NA_detection}")


# ## 1. 计算 Excitation PSF (Cylindrical Lens)
# 

# In[33]:


print("正在计算 Excitation PSF (Cylindrical Lens)...")
psf_excitation = focus_field_cylindrical(
    shape=shape,
    units=units,
    lam=lam_excitation,
    NA=NA_excitation,
    n0=n0,
    return_all_fields=False,
    n_integration_steps=200
)

print(f"Excitation PSF 形状: {psf_excitation.shape}")
print(f"Excitation PSF 最大值: {psf_excitation.max():.4f}")
print(f"Excitation PSF 最小值: {psf_excitation.min():.4f}")


# ## 2. 计算 Detection PSF (普通光束)
# 

# In[34]:


print("正在计算 Detection PSF (普通光束)...")
psf_detection = focus_field_beam(
    shape=shape,
    units=units,
    lam=lam_detection,
    NA=NA_detection,
    n0=n0,
    return_all_fields=False,
    n_integration_steps=200
)

print(f"Detection PSF 形状: {psf_detection.shape}")
print(f"Detection PSF 最大值: {psf_detection.max():.4f}")
print(f"Detection PSF 最小值: {psf_detection.min():.4f}")


# ## 3. 统一坐标系并计算 Effective PSF
# 

# In[35]:


# 统一坐标系：使用 Detection Objective 的坐标系
# Detection Objective 坐标系：
# - Z 轴：平行于 detection objective 的光轴（axis 0）
# - XY 平面：垂直于光轴（axis 1=Y, axis 2=X）
# 
# 当前状态：
# - Detection PSF: 已经在正确的坐标系中，Z 是光轴（axis 0），不需要旋转
# - Excitation PSF (lightsheet): 原本沿着 X 方向传播（axis 2），在 YZ 平面内很薄
#   - 需要旋转到 detection objective 坐标系
#   - 在 detection 坐标系中，excitation 应该沿着 Y 方向传播（XY 平面内，垂直于 Z 轴）
# 
# 坐标变换：
# - Excitation 原本：X 是传播方向（axis 2），YZ 是横向平面
# - 转换到 detection 坐标系：Y 是传播方向（axis 1），XZ 是横向平面
# - 方法：交换 X 和 Y 轴，即使用 np.swapaxes

print("正在统一坐标系到 Detection Objective 坐标系...")
print(f"Detection Objective: Z 是光轴（axis 0），XY 是横向平面（axis 1=Y, axis 2=X）")

# Detection PSF 已经在正确的坐标系中，不需要旋转
print(f"Detection PSF 形状: {psf_detection.shape} (已在 Detection Objective 坐标系中)")

# 将 Excitation PSF 转换到 Detection Objective 坐标系
# Excitation 原本：X 是传播方向（axis 2），需要转到 Y 方向（axis 1）
# 方法：交换 X 和 Y 轴
print(f"Excitation PSF 原始形状: {psf_excitation.shape}")
print("正在将 Excitation PSF 从 (X 传播方向) 转换到 (Y 传播方向)...")
# 交换 axis 1 (Y) 和 axis 2 (X)
psf_excitation_det_coords = np.swapaxes(psf_excitation, 0, 1)
# psf_excitation_det_coords = psf_excitation
print(f"转换后 Excitation PSF 形状: {psf_excitation_det_coords.shape}")
print("说明：Excitation PSF 现在在 Detection Objective 坐标系中，Y 是传播方向，XZ 是横向平面")

# 现在在统一的坐标系中计算 Effective PSF
# Detection PSF: Z 是光轴
# Excitation PSF: Y 是传播方向，与 Z 轴正交
psf_effective = psf_excitation_det_coords * psf_detection

# 更新变量名，使用统一坐标系后的数据
psf_excitation = psf_excitation_det_coords

print(f"Effective PSF 形状: {psf_effective.shape}")
print(f"Effective PSF 最大值: {psf_effective.max():.4f}")
print(f"Effective PSF 最小值: {psf_effective.min():.4f}")

# 保存 Effective PSF 到文件
import os
from tifffile import imwrite

psfdata_dir = 'psfdata'
os.makedirs(psfdata_dir, exist_ok=True)
output_file = os.path.join(psfdata_dir, 'cylindrical_lightsheet_effective_psf.tif')
print(f"正在保存 Effective PSF 到: {output_file}")
imwrite(output_file, psf_effective.astype(np.float32),imagej=True)
print(f"保存完成！文件: {output_file}")

assert 1==2


# ## 4. 可视化 PSF - 3x2 组合可视化
# 
# 创建一个 3x2 的子图布局：
# - 第1行：Excitation PSF（左侧3D Volume，右侧3个2D截面：XY, XZ, YZ）
# - 第2行：Detection PSF（左侧3D Volume，右侧3个2D截面：XY, XZ, YZ）
# - 第3行：Effective PSF（左侧3D Volume，右侧3个2D截面：XY, XZ, YZ）
# 

# In[36]:


# 导入 3D 可视化所需的库
import plotly.graph_objects as go
import plotly.io as pio

# 设置 plotly 默认主题为暗色
pio.templates.default = 'plotly_dark'

# 配置 plotly 渲染器，使用 browser 渲染器在浏览器中打开 HTML
pio.renderers.default = 'browser'

print("3D 可视化库导入完成")
print(f"Plotly 渲染器: {pio.renderers.default}")
print("提示: 所有图表将在浏览器中打开（HTML 格式）")


# ### 7.1 使用 Plotly 的 3D Volume 可视化
# 

# In[44]:


def visualize_micro_3D(data, title="3D Volume Visualization", isomin=None, opacity=0.05, surface_count=21, colorscale='turbo_r', downsample=1):
    """
    使用 Plotly Volume 可视化 3D 数据
    
    坐标系：统一使用 Detection Objective 坐标系
    - Z 轴：detection objective 的光轴（axis 0）
    - XY 平面：垂直于光轴（axis 1=Y, axis 2=X）

    参数:
    - data: PSF 数据，形状为 (Nz, Ny, Nx)，即 (Z, Y, X)
    - title: 图表标题
    - isomin: 最小显示值（如果为 None，则使用最大值的 0.8）
    - opacity: 透明度
    - surface_count: 等值面数量
    - colorscale: 颜色方案
    - downsample: 降采样因子
    """
    # 降采样以提高性能
    if downsample > 1:
        data_downsampled = data[::downsample, ::downsample, ::downsample]
        print(f"降采样后形状: {data_downsampled.shape}")
    else:
        data_downsampled = data

    # 数据格式是 (Z, Y, X)，直接使用，不需要转置
    # data_downsampled 形状为 (Nz, Ny, Nx)，即 (Z, Y, X)
    data_xyz = data_downsampled

    Nz, Ny, Nx = data_xyz.shape

    # 创建坐标网格（使用物理坐标）
    # 注意：数据是 (Z, Y, X) 格式
    z_coords = np.arange(Nz) * units[2] - (Nz // 2) * units[2]  # Z 轴（光轴）
    y_coords = np.arange(Ny) * units[1] - (Ny // 2) * units[1]  # Y 轴
    x_coords = np.arange(Nx) * units[0] - (Nx // 2) * units[0]  # X 轴

    # 创建网格，data_xyz 形状为 (Nz, Ny, Nx)
    Z, Y, X = np.mgrid[
        0:data_xyz.shape[0],  # Z 维度
        0:data_xyz.shape[1],  # Y 维度
        0:data_xyz.shape[2]   # X 维度
    ]

    # 将像素索引转换为物理坐标
    Z = z_coords[Z]  # Z 轴（光轴）
    Y = y_coords[Y]  # Y 轴
    X = x_coords[X]  # X 轴

    # 设置 isomin
    if isomin is None:
        isomin = data_xyz.max() * 0.8

    # 创建 Volume 图
    vol = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=data_xyz.flatten(),
            opacity=opacity,
            surface_count=surface_count,
            colorscale=colorscale,
            isomin=isomin,
        )
    )

    vol.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (μm) - Detection Objective 光轴',
            yaxis_title='Y (μm) - Detection Objective 坐标系',
            zaxis_title='Z (μm) - Detection Objective 坐标系',
            aspectmode='data'
        ),
        height=1200
    )

    return vol

def visualize_all_psf_combined(psf_excitation, psf_detection, psf_effective, 
                                title="PSF 组合可视化", isomin=None, opacity=0.05,
                                surface_count=21, colorscale_3d='turbo_r', colorscale_2d='hot',
                                downsample=2):
    """
    创建 4x3 子图布局（每行 4 个 subplot，共 3 行）：
    - 第1行：Excitation PSF（3D Volume, YZ截面, XY截面, XZ截面）
    - 第2行：Detection PSF（3D Volume, YZ截面, XY截面, XZ截面）
    - 第3行：Effective PSF（3D Volume, YZ截面, XY截面, XZ截面）
    
    坐标系：统一使用 Detection Objective 坐标系
    - Z 轴：detection objective 的光轴（axis 0）
    - XY 平面：垂直于光轴（axis 1=Y, axis 2=X）
    - 所有 PSF 数据形状为 (Nz, Ny, Nx)，即 (Z, Y, X)
    """
    fig = go.Figure()
    
    psf_list = [psf_excitation, psf_detection, psf_effective]
    psf_names = ['Excitation PSF', 'Detection PSF', 'Effective PSF']
    
    # 处理每个 PSF
    for row_idx, (psf, psf_name) in enumerate(zip(psf_list, psf_names), 1):
        # 降采样
        if downsample > 1:
            data_downsampled = psf[::downsample, ::downsample, ::downsample]
        else:
            data_downsampled = psf
        
        # 数据格式是 (Z, Y, X)，直接使用，不需要转置
        # data_downsampled 形状为 (Nz, Ny, Nx)，即 (Z, Y, X)
        data_xyz = data_downsampled
        data_original = data_downsampled
        
        Nz, Ny, Nx = data_xyz.shape
        
        # 创建坐标
        # 注意：数据是 (Z, Y, X) 格式
        z_coords = np.arange(Nz) * units[2] - (Nz // 2) * units[2]  # Z 轴（光轴）
        y_coords = np.arange(Ny) * units[1] - (Ny // 2) * units[1]  # Y 轴
        x_coords = np.arange(Nx) * units[0] - (Nx // 2) * units[0]  # X 轴
        
        # 提取中心切片
        # data_original 形状为 (Nz, Ny, Nx)，即 (Z, Y, X)
        center_z = Nz // 2  # Z 方向的中心（光轴中心）
        center_y = Ny // 2  # Y 方向的中心
        center_x = Nx // 2  # X 方向的中心
        
        # 在统一坐标系中提取切片：
        # - YZ 平面：固定 X，取 (Z, Y)，即 data_original[:, :, center_x]
        # - XY 平面：固定 Z（光轴），取 (Y, X)，即 data_original[center_z, :, :]
        # - XZ 平面：固定 Y，取 (Z, X)，即 data_original[:, center_y, :]
        slice_yz = data_original[:, :, center_x]  # YZ 平面，形状 (Nz, Ny) = (Z, Y)
        slice_xy = data_original[center_z, :, :]  # XY 平面，形状 (Ny, Nx) = (Y, X)
        slice_xz = data_original[:, center_y, :]  # XZ 平面，形状 (Nz, Nx) = (Z, X)
        
        # 创建 3D 坐标网格
        # data_xyz 形状为 (Nz, Ny, Nx)，即 (Z, Y, X)
        Z, Y, X = np.mgrid[
            0:data_xyz.shape[0],  # Z 维度
            0:data_xyz.shape[1],  # Y 维度
            0:data_xyz.shape[2]   # X 维度
        ]
        Z = z_coords[Z]  # Z 轴（光轴）
        Y = y_coords[Y]  # Y 轴
        X = x_coords[X]  # X 轴
        
        # 设置 isomin
        if isomin is None:
            isomin_val = data_xyz.max() * 0.8
        else:
            isomin_val = isomin
        
        # 计算每行的垂直位置（从下往上：row 1 在底部，row 3 在顶部）
        row_height = 1.0 / 3.0
        y_bottom = (row_idx - 1) * row_height
        y_top = row_idx * row_height
        
        # 4x3 布局：每行 4 个 subplot，每个占据 1/4 宽度
        col_width = 1.0 / 4.0
        
        # 添加 3D Volume（第1列，占据 0-1/4）
        fig.add_trace(go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=data_xyz.flatten(),
            opacity=opacity,
            surface_count=surface_count,
            colorscale=colorscale_3d,
            isomin=isomin_val,
            scene=f'scene{row_idx}',
            showscale=(row_idx == 1),  # 只在第一行显示颜色条
            name=f'{psf_name} 3D'
        ))
        
        # 添加三个 2D Heatmap（第2、3、4列，每个占据一个独立的 subplot）
        
        # YZ 平面（第2列，占据 1/4-2/4）
        vmax_yz = slice_yz.max()
        fig.add_trace(go.Heatmap(
            z=slice_yz.T,  # 转置：从 (Nz, Ny) 到 (Ny, Nz)
            x=z_coords,
            y=y_coords,
            colorscale=colorscale_2d,
            zmax=vmax_yz,
            xaxis=f'x{row_idx*10+1}',
            yaxis=f'y{row_idx*10+1}',
            showscale=(row_idx == 1),  # 只在第一行显示颜色条
            name=f'{psf_name} YZ'
        ))
        
        # XY 平面（第3列，占据 2/4-3/4）
        vmax_xy = slice_xy.max()
        fig.add_trace(go.Heatmap(
            z=slice_xy,
            x=y_coords,
            y=x_coords,
            colorscale=colorscale_2d,
            zmax=vmax_xy,
            xaxis=f'x{row_idx*10+2}',
            yaxis=f'y{row_idx*10+2}',
            showscale=False,
            name=f'{psf_name} XY'
        ))
        
        # XZ 平面（第4列，占据 3/4-4/4）
        vmax_xz = slice_xz.max()
        fig.add_trace(go.Heatmap(
            z=slice_xz,
            x=z_coords,
            y=x_coords,
            colorscale=colorscale_2d,
            zmax=vmax_xz,
            xaxis=f'x{row_idx*10+3}',
            yaxis=f'y{row_idx*10+3}',
            showscale=False,
            name=f'{psf_name} XZ'
        ))
    
    # 配置布局
    layout_dict = {
        'title': dict(text=title, font=dict(size=20)),
        'height': 1800,
        'width': 2400,
        'template': 'plotly_dark'
    }
    
    # 为每一行配置 3D scene 和 2D 轴（4x3 布局）
    for row in range(1, 4):
        row_height = 1.0 / 3.0
        y_bottom = (row - 1) * row_height
        y_top = row * row_height
        col_width = 1.0 / 4.0
        
        # 3D scene（第1列，占据 0-1/4）
        # 坐标系：统一使用 Detection Objective 坐标系
        # Z 轴：detection objective 的光轴
        # XY 平面：垂直于光轴
        layout_dict[f'scene{row}'] = dict(
            domain=dict(x=[0, col_width], y=[y_bottom, y_top]),
            xaxis_title='X (μm) - Detection Objective 光轴',
            yaxis_title='Y (μm) - Detection Objective',
            zaxis_title='Z (μm) - Detection Objective',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
        
        # 三个 2D 图的轴配置（第2、3、4列，每个占据一个独立的 subplot）
        # 使用 scaleanchor 和 scaleratio 确保每个切片都是正方形
        col_margin = 0.02  # 列之间的边距
        square_size = min(col_width - 2*col_margin, row_height - 2*col_margin)
        
        # YZ 平面（第2列，占据 1/4-2/4）
        yz_x_center = col_width + col_width / 2
        yz_y_center = (y_bottom + y_top) / 2
        
        # YZ 平面（第2列）- 正方形
        # 显示 (Z, Y)：x 轴是 Z（光轴），y 轴是 Y
        # 根据 3D scene 映射：x=Z, y=Y, z=X
        layout_dict[f'xaxis{row*10+1}'] = dict(
            domain=[yz_x_center - square_size/2, yz_x_center + square_size/2],
            anchor=f'y{row*10+1}',
            title='Z (μm)',  # x 轴显示 Z（光轴）
            scaleanchor=f'y{row*10+1}',
            scaleratio=1.0
        )
        layout_dict[f'yaxis{row*10+1}'] = dict(
            domain=[yz_y_center - square_size/2, yz_y_center + square_size/2],
            anchor=f'x{row*10+1}',
            title='Y (μm)'  # y 轴显示 Y
        )
        
        # XY 平面（第3列）- 正方形
        # 显示 (Y, X)：x 轴是 Y，y 轴是 X
        # 根据 3D scene 映射：x=Z, y=Y, z=X
        xy_x_center = 2 * col_width + col_width / 2
        xy_y_center = (y_bottom + y_top) / 2
        layout_dict[f'xaxis{row*10+2}'] = dict(
            domain=[xy_x_center - square_size/2, xy_x_center + square_size/2],
            anchor=f'y{row*10+2}',
            title='Y (μm)',  # x 轴显示 Y
            scaleanchor=f'y{row*10+2}',
            scaleratio=1.0
        )
        layout_dict[f'yaxis{row*10+2}'] = dict(
            domain=[xy_y_center - square_size/2, xy_y_center + square_size/2],
            anchor=f'x{row*10+2}',
            title='X (μm)'  # y 轴显示 X
        )
        
        # XZ 平面（第4列）- 正方形
        # 显示 (Z, X)：x 轴是 Z（光轴），y 轴是 X
        # 根据 3D scene 映射：x=Z, y=Y, z=X
        xz_x_center = 3 * col_width + col_width / 2
        xz_y_center = (y_bottom + y_top) / 2
        layout_dict[f'xaxis{row*10+3}'] = dict(
            domain=[xz_x_center - square_size/2, xz_x_center + square_size/2],
            anchor=f'y{row*10+3}',
            title='Z (μm)',  # x 轴显示 Z（光轴）
            scaleanchor=f'y{row*10+3}',
            scaleratio=1.0
        )
        layout_dict[f'yaxis{row*10+3}'] = dict(
            domain=[xz_y_center - square_size/2, xz_y_center + square_size/2],
            anchor=f'x{row*10+3}',
            title='X (μm)'  # y 轴显示 X
        )
    
    fig.update_layout(**layout_dict)
    
    return fig

print("Plotly 3D Volume 可视化函数已定义")
print("Plotly 4x3 组合可视化函数已定义")


# In[45]:


# 生成 4x3 组合可视化
print("正在生成 PSF 4x3 组合可视化...")
print(f"NA_exc={NA_excitation}, NA_det={NA_detection}")

fig = visualize_all_psf_combined(
    psf_excitation,
    psf_detection,
    psf_effective,
    title=f'PSF 组合可视化<br>NA_exc={NA_excitation}, NA_det={NA_detection}',
    isomin=None,
    opacity=0.05,
    surface_count=21,
    colorscale_3d='turbo_r',
    colorscale_2d='hot',
    downsample=2
)

fig.show()
print("可视化完成！")


