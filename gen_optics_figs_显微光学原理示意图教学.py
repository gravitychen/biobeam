import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    'font.family': 'Microsoft YaHei',
    'font.size': 10,
    'figure.facecolor': 'white',
    'axes.unicode_minus': False,
})

OUT = r'C:\Code\biobeam\figures'

# ─────────────────────────────────────────────────────────────────────────────
# 通用辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def vline(ax, x, y0, y1, color='#aaa', lw=1.2, ls='--', zorder=2):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, ls=ls, zorder=zorder)

def arrow(ax, x0, y0, x1, y1, color='black', lw=1.5, ms=12, zorder=3):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=ms),
                zorder=zorder)

def draw_biconvex(ax, x, h, color='#3B82F6', lw=2.5):
    """画双凸透镜符号（垂直线段 + 两端箭头帽）"""
    ax.plot([x, x], [-h, h], color=color, lw=lw, solid_capstyle='round', zorder=4)
    ax.annotate('', xy=(x,  h+0.15), xytext=(x,  h),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=14), zorder=4)
    ax.annotate('', xy=(x, -h-0.15), xytext=(x, -h),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=14), zorder=4)

def draw_rect_lens(ax, x0, x1, h, color='#93C5FD', ec='#3B82F6', label='', lw=1.5):
    """画矩形厚透镜"""
    rect = mpatches.FancyBboxPatch(
        (x0, -h), x1-x0, 2*h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=ec, lw=lw, zorder=3, alpha=0.7)
    ax.add_patch(rect)
    if label:
        ax.text((x0+x1)/2, 0, label, ha='center', va='center',
                fontsize=8.5, color='#1E3A8A', zorder=5)

def optical_axis(ax, x0, x1, y=0):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='->', color='#777', lw=1.1,
                                mutation_scale=10), zorder=1)

BOX = dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none', alpha=0.88)

# ═════════════════════════════════════════════════════════════════════════════
# 图 1：薄透镜三条特殊光线 + 各平面标注
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6.5))
ax.set_xlim(-7.2, 7.2)
ax.set_ylim(-2.8, 3.6)
ax.set_aspect('equal')
ax.axis('off')

f = 3.5
ray_y = 1.1   # 入射光线高度

# 光轴
optical_axis(ax, -7.0, 7.0)
ax.text(7.1, 0, 'z', va='center', ha='left', fontsize=10, color='#666')

# 薄透镜
draw_biconvex(ax, 0, 1.6, color='#3B82F6', lw=3)

# 前焦点 F  后焦点 F'
Fl, Fr = -f, f
vline(ax, Fl, -2.0, 2.6, color='#D97706', lw=1.8, ls='--')
vline(ax, Fr, -2.0, 2.6, color='#059669', lw=1.8, ls='--')
vline(ax,  0, -2.2, 3.2, color='#7C3AED', lw=2.2, ls='-')

ax.plot(Fl, 0, 'o', color='#D97706', ms=7, zorder=6)
ax.plot(Fr, 0, 'o', color='#059669', ms=7, zorder=6)

# 光线 1：平行入射 → 折射后过 F'
c1 = '#EF4444'
ax.plot([-7.0, 0], [ray_y, ray_y], color=c1, lw=1.8)
arrow(ax, -7.0, ray_y, -3.5, ray_y, color=c1, lw=1.8)
# 折射后过 F'=(Fr, 0)：斜率 = (0-ray_y)/(Fr-0)
m1 = -ray_y / Fr
ax.plot([0, 7.0], [ray_y, ray_y + m1*7.0], color=c1, lw=1.8)
arrow(ax, 0, ray_y, 3.5, ray_y + m1*3.5, color=c1, lw=1.8)

# 光线 2：过中心不偏折
c2 = '#F97316'
m2 = -ray_y / 7.0
ax.plot([-7.0, 7.0], [ray_y, ray_y + m2*14], color=c2, lw=1.8)
arrow(ax, -7.0, ray_y, -3.0, ray_y + m2*4.0, color=c2, lw=1.8)
arrow(ax, 0, 0, 3.5, m2*10.5, color=c2, lw=1.8)

# 光线 3：从 F 出发 → 折射后平行
c3 = '#6366F1'
m3 = ray_y / (0 - Fl)   # 斜率（F 到透镜）
ax.plot([Fl, 0], [0, ray_y], color=c3, lw=1.8)
arrow(ax, Fl, 0, (Fl+0)/2, ray_y/2, color=c3, lw=1.8)
ax.plot([0, 7.0], [ray_y, ray_y], color=c3, lw=1.8)
arrow(ax, 0, ray_y, 3.5, ray_y, color=c3, lw=1.8)

# 像点（三线交汇）
u_obj = 7.0   # 物距（取正值，物在左 7）
v_img = 1/(1/f + 1/(-u_obj))   # 薄透镜公式 1/v - 1/u = 1/f，u<0
# 用几何：光线1在像面的 y = ray_y + m1*v_img
y_img = ray_y + m1*v_img
ax.plot(v_img, y_img, '*', ms=16, color='gold',
        markeredgecolor='#92400E', markeredgewidth=0.8, zorder=7)

# ── 标注 ──────────────────────────────────────────────────────────────────
ax.text(Fl,  2.65, '前焦面\n(Front Focal Plane, FFP)',
        ha='center', va='bottom', fontsize=8.5, color='#92400E', bbox=BOX)
ax.text(Fr,  2.65, '后焦面\n(Back Focal Plane, BFP)',
        ha='center', va='bottom', fontsize=8.5, color='#065F46', bbox=BOX)
ax.text(0,   3.22, '前主面 H = 后主面 H′\n(Front/Back Principal Plane, 薄透镜时重合)',
        ha='center', va='bottom', fontsize=8.5, color='#5B21B6', bbox=BOX)

ax.text(Fl-0.12, -0.22, 'F  前焦点\n(Front Focal Point)',
        ha='right', va='top', fontsize=8, color='#92400E')
ax.text(Fr+0.12, -0.22, "F′  后焦点\n(Back Focal Point)",
        ha='left',  va='top', fontsize=8, color='#065F46')

ax.text(v_img+0.15, y_img+0.05, '像点 (Image Point)', ha='left',
        va='bottom', fontsize=8.5, color='#78350F', bbox=BOX)

# 焦距双箭头
def dim_arrow(ax, x0, x1, y, label, color):
    ax.annotate('', xy=(x1,y), xytext=(x0,y),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.3,
                                mutation_scale=10))
    ax.text((x0+x1)/2, y-0.18, label, ha='center', va='top',
            fontsize=8.5, color=color, bbox=BOX)

dim_arrow(ax, Fl, 0, -2.0, 'f  前焦距 (front focal length)', '#B45309')
dim_arrow(ax,  0, Fr, -2.55, "f ′  后焦距 (back focal length)", '#065F46')

ax.text(0.15, -1.85, '薄透镜\n(Thin Lens)\nH = H′', ha='left', va='top',
        fontsize=8.5, color='#3B82F6', bbox=BOX)

leg = [Line2D([0],[0],color=c1,lw=2,label='① 平行入射 → 过 F′'),
       Line2D([0],[0],color=c2,lw=2,label='② 过中心 → 直线不偏折'),
       Line2D([0],[0],color=c3,lw=2,label='③ 过 F → 折射后平行出射')]
ax.legend(handles=leg, loc='lower left', fontsize=8.5,
          framealpha=0.92, edgecolor='#ddd')

ax.set_title('图1  薄透镜 (Thin Lens)：三条特殊光线与各平面定义',
             fontsize=12, fontweight='bold', pad=6)
plt.tight_layout()
plt.savefig(f'{OUT}/optics_fig1_thin_lens.png', dpi=160,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 1 done.')

# ═════════════════════════════════════════════════════════════════════════════
# 图 2：厚透镜 / 显微物镜 — 主面 H H' 分离
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6.0))
ax.set_xlim(-7.5, 7.5)
ax.set_ylim(-2.6, 3.2)
ax.set_aspect('equal')
ax.axis('off')

# 参数
fL = 3.0   # 前焦距（H 到 F）
fR = 3.0   # 后焦距（H' 到 F'）
Hx  = -0.8  # 前主面 H 位置
Hpx =  0.8  # 后主面 H' 位置
Fx  = Hx  - fL   # 前焦点 = -3.8
Fpx = Hpx + fR   # 后焦点 =  3.8
lens_x0, lens_x1 = -2.0, 2.0  # 厚透镜物理范围

# 光轴
optical_axis(ax, -7.2, 7.2)
ax.text(7.3, 0, 'z', va='center', ha='left', fontsize=10, color='#666')

# 厚透镜矩形
draw_rect_lens(ax, lens_x0, lens_x1, 1.5, color='#BFDBFE', ec='#3B82F6',
               label='厚透镜 / 物镜\n(Thick Lens / Objective)', lw=2)

# 主面
vline(ax, Hx,  -2.0, 2.8, color='#7C3AED', lw=2.0, ls='-.')
vline(ax, Hpx, -2.0, 2.8, color='#DB2777', lw=2.0, ls='-.')

# 焦面
vline(ax, Fx,  -1.8, 2.6, color='#D97706', lw=1.8, ls='--')
vline(ax, Fpx, -1.8, 2.6, color='#059669', lw=1.8, ls='--')

ax.plot(Fx,  0, 'o', color='#D97706', ms=7, zorder=6)
ax.plot(Fpx, 0, 'o', color='#059669', ms=7, zorder=6)

# 光线（平行入射 → 过 H' → 过 F'）
ray_y2 = 1.0
c1, c3 = '#EF4444', '#6366F1'

# 光线 A：平行入射，到 H 之前保持平行，H' 之后过 F'
ax.plot([-7.0, Hx], [ray_y2, ray_y2], color=c1, lw=1.8)
arrow(ax, -7.0, ray_y2, -4.5, ray_y2, color=c1, lw=1.8)
# H' 到 F' 方向
m_A = (0 - ray_y2) / (Fpx - Hpx)
ax.plot([Hpx, 7.0], [ray_y2, ray_y2 + m_A*(7.0-Hpx)], color=c1, lw=1.8)
arrow(ax, Hpx, ray_y2, Fpx+0.3, ray_y2+m_A*0.3, color=c1, lw=1.8)
# 透镜内部虚线连接
ax.plot([Hx, Hpx], [ray_y2, ray_y2], color=c1, lw=1.0, ls=':')

# 光线 B：从 F 出发 → 到 H 后水平
m_B = ray_y2 / (Hx - Fx)
ax.plot([Fx, Hx], [0, ray_y2], color=c3, lw=1.8)
arrow(ax, Fx, 0, (Fx+Hx)/2, ray_y2/2, color=c3, lw=1.8)
ax.plot([Hpx, 7.0], [ray_y2, ray_y2], color=c3, lw=1.8)
arrow(ax, Hpx, ray_y2, Hpx+2, ray_y2, color=c3, lw=1.8)
ax.plot([Hx, Hpx], [ray_y2, ray_y2], color=c3, lw=1.0, ls=':')

# 标注
ax.text(Hx,  2.85, '前主面 H\n(Front Principal\nPlane, FPP)',
        ha='center', va='bottom', fontsize=8.5, color='#5B21B6', bbox=BOX)
ax.text(Hpx, 2.85, '后主面 H′\n(Back Principal\nPlane, RPP)',
        ha='center', va='bottom', fontsize=8.5, color='#9D174D', bbox=BOX)
ax.text(Fx,  2.65, '前焦面\n(FFP)',
        ha='center', va='bottom', fontsize=8.5, color='#92400E', bbox=BOX)
ax.text(Fpx, 2.65, '后焦面\n(BFP)',
        ha='center', va='bottom', fontsize=8.5, color='#065F46', bbox=BOX)

ax.text(Fx-0.1,  -0.25, 'F', ha='right', va='top', fontsize=9, color='#B45309', fontweight='bold')
ax.text(Fpx+0.1, -0.25, "F′", ha='left', va='top', fontsize=9, color='#059669', fontweight='bold')

# 焦距标注
dim_arrow(ax, Hx,  Fx,  -1.9, 'f（从 H 量）', '#B45309')
dim_arrow(ax, Hpx, Fpx, -2.45, "f ′（从 H′ 量）", '#065F46')

# H 与 H' 分离距离
dim_arrow(ax, Hx, Hpx, -1.2, 'H 与 H′ 的间距\n(principal plane separation)', '#7C3AED')

# 注释
ax.text(-6.8, -2.4,
        '关键：焦距 f 和 f′ 从主面 H / H′ 量起，\n而不是从透镜的物理端面量起。',
        ha='left', va='bottom', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFBEB', edgecolor='#F59E0B', lw=1))

leg2 = [Line2D([0],[0],color=c1,lw=2,label='平行入射 → H/H′等高处折射 → 过 F′'),
        Line2D([0],[0],color=c3,lw=2,label='过 F 出发 → H/H′等高处折射 → 平行出射')]
ax.legend(handles=leg2, loc='lower right', fontsize=8.5,
          framealpha=0.92, edgecolor='#ddd')

ax.set_title('图2  厚透镜 (Thick Lens)：前主面 H 与后主面 H′ 分离',
             fontsize=12, fontweight='bold', pad=6)
plt.tight_layout()
plt.savefig(f'{OUT}/optics_fig2_thick_lens.png', dpi=160,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 2 done.')

# ═════════════════════════════════════════════════════════════════════════════
# 图 3：无限远校正显微镜 — 完整光路
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 6.5))
ax.set_xlim(-0.5, 16.5)
ax.set_ylim(-2.8, 3.4)
ax.set_aspect('equal')
ax.axis('off')

# ── 位置参数（单位任意）────────────────────────────────────────
x_sample  = 0.8        # 样品 / 前焦面
x_obj_l   = 1.4        # 物镜左端
x_obj_r   = 4.0        # 物镜右端
x_bfp     = 5.5        # 后焦面 / 后瞳面（BFP / BPP）
x_tl_l    = 9.5        # 管镜左端
x_tl_r    = 10.8       # 管镜右端
x_image   = 12.5       # 像面（前焦面 of tube lens）

# 光轴
ax.annotate('', xy=(16.3, 0), xytext=(-0.3, 0),
            arrowprops=dict(arrowstyle='->', color='#777', lw=1.1, mutation_scale=10))
ax.text(16.4, 0, 'z', va='center', ha='left', fontsize=10, color='#666')

# 样品（点光源）
ax.plot(x_sample, 0.0, '*', ms=16, color='gold',
        markeredgecolor='#92400E', markeredgewidth=1, zorder=7)
ax.plot(x_sample, 0.9, 'o', ms=6, color='#EF4444', zorder=7)  # 离轴点

# 物镜
draw_rect_lens(ax, x_obj_l, x_obj_r, 1.6, color='#BFDBFE', ec='#3B82F6',
               label='物镜\n(Objective)', lw=2)

# 管镜
draw_rect_lens(ax, x_tl_l, x_tl_r, 1.3, color='#D1FAE5', ec='#059669',
               label='管镜\n(Tube Lens)', lw=2)

# 各平面竖线
vline(ax, x_sample, -2.0, 2.8, color='#D97706', lw=2.0, ls='--')
vline(ax, x_bfp,   -2.0, 2.8, color='#059669', lw=2.0, ls='--')
vline(ax, x_image, -2.0, 2.8, color='#DC2626', lw=2.0, ls='--')

# ── 三束平行光（样品轴上点 → 物镜准直）──────────────────────────
ys_parallel = [-1.1, 0.0, 1.1]
colors_p = ['#60A5FA', '#3B82F6', '#1D4ED8']
for yp, cp in zip(ys_parallel, colors_p):
    # 样品到物镜：会聚到样品点 → 角度 = yp / (x_obj_l - x_sample)
    m_in = yp / (x_obj_l - x_sample) if (x_obj_l - x_sample) != 0 else 0
    y_at_obj = yp   # 在物镜入射面的高度（近似）
    # 物镜准直后：平行光
    ax.plot([x_sample, x_obj_l], [0, -yp*(x_obj_l-x_sample)/(x_obj_r-x_sample)],
            color=cp, lw=1.5, alpha=0.7)
    ax.plot([x_obj_r, x_tl_l], [yp, yp], color=cp, lw=1.8, alpha=0.9)
    arrow(ax, x_obj_r+(x_tl_l-x_obj_r)*0.3, yp,
               x_obj_r+(x_tl_l-x_obj_r)*0.6, yp, color=cp, lw=1.8, ms=10)

# 离轴点的平行光束（不同角度）
y_off = 0.9
c_off = '#F97316'
ys_off = [-1.0, 0.0, 1.0]
for yo in ys_off:
    ax.plot([x_sample, x_obj_l], [y_off, yo + (y_off-yo)*(x_obj_l-x_sample)/(x_obj_r-x_sample)*0],
            color=c_off, lw=1.2, ls=':', alpha=0.55)
    ax.plot([x_obj_r, x_tl_l], [yo, yo], color=c_off, lw=1.2, ls=':', alpha=0.55)

# 管镜聚焦到像面
for yp, cp in zip(ys_parallel, colors_p):
    m_tl = -yp / (x_image - x_tl_r)
    ax.plot([x_tl_r, x_image], [yp, 0], color=cp, lw=1.8, alpha=0.9)

# ── 平面标注 ──────────────────────────────────────────────────
def plane_label(ax, x, top_text, bot_text, top_color, bot_color=None):
    bot_color = bot_color or top_color
    ax.text(x, 2.88, top_text, ha='center', va='bottom',
            fontsize=8.2, color=top_color, bbox=BOX)
    ax.text(x, -2.1, bot_text, ha='center', va='top',
            fontsize=7.5, color=bot_color, bbox=BOX)

plane_label(ax, x_sample,
            '前焦面 (FFP)\n= 样品面 (Sample Plane)',
            'Object at FFP\n→ 出射平行光',
            '#92400E')

plane_label(ax, x_bfp,
            '后焦面 (BFP)\n= 后瞳面 (Back Pupil Plane, BPP)\n= 放 SLM / 相位板的位置',
            '空间频谱面\n(Fourier / Frequency Plane)',
            '#065F46')

plane_label(ax, x_image,
            '像面 (Image Plane)\n= 管镜前焦面 (FFP of TL)',
            '相机 (Camera)\n安装于此',
            '#991B1B')

# 平行光区域标注
mid_par = (x_obj_r + x_tl_l) / 2
ax.text(mid_par, 1.65,
        '平行光区 (Collimated / Infinity Space)\n物镜 BFP ↔ 管镜 FFP',
        ha='center', va='bottom', fontsize=8.2,
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#F0FDF4',
                  edgecolor='#6EE7B7', lw=1.2))
ax.annotate('', xy=(x_tl_l-0.1, 1.55), xytext=(x_obj_r+0.1, 1.55),
            arrowprops=dict(arrowstyle='<->', color='#10B981', lw=1.3, mutation_scale=10))

# SLM 标注
ax.annotate('SLM / 相位板\n(Spatial Light Modulator)\n插入此处操控瞳函数 P(kx,ky)',
            xy=(x_bfp, 0), xytext=(x_bfp+1.6, -2.0),
            fontsize=8, color='#065F46',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D1FAE5', edgecolor='#059669', lw=1),
            arrowprops=dict(arrowstyle='->', color='#059669', lw=1.3))

# 图例
leg3 = [Line2D([0],[0],color='#3B82F6',lw=2.5,label='轴上点光源的平行光束'),
        Line2D([0],[0],color='#F97316',lw=2,ls=':',label='离轴点光源（不同角度平行光）')]
ax.legend(handles=leg3, loc='upper left', fontsize=8.5,
          framealpha=0.92, edgecolor='#ddd', bbox_to_anchor=(0.0, 0.98))

ax.set_title(
    '图3  无限远校正显微镜 (Infinity-Corrected Microscope) 完整光路\n'
    '样品 (Sample) → 物镜 (Objective) → 后瞳面/BFP → 管镜 (Tube Lens) → 像面 (Image Plane)',
    fontsize=11, fontweight='bold', pad=6)

plt.tight_layout()
plt.savefig(f'{OUT}/optics_fig3_microscope.png', dpi=160,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 3 done.')

# ═════════════════════════════════════════════════════════════════════════════
# 图 4：后瞳面 (Back Pupil Plane) 上的瞳函数 P(kx, ky)
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('图4  后瞳面 (Back Pupil Plane) 上的瞳函数 P(kx, ky) — 四种光束',
             fontsize=12, fontweight='bold')

N = 300
ks = np.linspace(-1, 1, N)
KX, KY = np.meshgrid(ks, ks)
KR = np.sqrt(KX**2 + KY**2)
PHI = np.arctan2(KY, KX)

configs = [
    ('DSLM / 高斯光束\n(Gaussian Beam)', KR <= 0.5, 'Blues'),
    ('Bessel 光束\n(Bessel Beam)', (KR >= 0.42) & (KR <= 0.55), 'Greens'),
    ('晶格光片 LLS\n(Lattice Light-Sheet, kpoints=6)',
     None, 'Reds'),
]

NA1_l, NA2_l = 0.42, 0.55
k_c = (NA1_l + NA2_l) / 2
sigma_k = 0.015
ring_l = (KR >= NA1_l) & (KR <= NA2_l)
ts = np.pi * (0.5 + 2.0/6 * np.arange(6))
kxs_l = k_c * np.cos(ts)
kys_l = k_c * np.sin(ts)
amp_l = np.zeros((N, N))
for kxi, kyi in zip(kxs_l, kys_l):
    amp_l += np.exp(-((KX-kxi)**2 + (KY-kyi)**2) / (2*sigma_k**2))
amp_l *= ring_l
if amp_l.max() > 0:
    amp_l /= amp_l.max()

pupils = [
    (KR <= 0.5).astype(float),
    ((KR >= 0.42) & (KR <= 0.55)).astype(float),
    amp_l,
]
cmaps   = ['Blues', 'Greens', 'Reds']
titles  = [
    'DSLM / 高斯光束\n(Gaussian Beam)\nNA = 0.5',
    'Bessel 光束\n(Bessel Beam)\nNA₁=0.42, NA₂=0.55',
    '晶格光片 LLS\n(Lattice Light-Sheet)\nkpoints=6, σ=0.015',
]

theta_circ = np.linspace(0, 2*np.pi, 300)

for ax_p, P, cmap, title in zip(axes, pupils, cmaps, titles):
    ax_p.imshow(P, cmap=cmap, origin='lower',
                extent=[-1, 1, -1, 1], vmin=0, vmax=1)
    # 外圈（最大 NA 边界）
    ax_p.plot(np.cos(theta_circ), np.sin(theta_circ),
              'k--', lw=1.2, alpha=0.5, label='NA 边界')
    ax_p.set_xlabel('$k_x$ / $k_{max}$', fontsize=9)
    ax_p.set_ylabel('$k_y$ / $k_{max}$', fontsize=9)
    ax_p.set_title(title, fontsize=9.5)
    ax_p.set_aspect('equal')
    # 轴标签
    ax_p.axhline(0, color='white', lw=0.6, alpha=0.4)
    ax_p.axvline(0, color='white', lw=0.6, alpha=0.4)

# 在 BFP 图右侧加说明
axes[2].text(1.12, 0,
    '后瞳面上\n每个点 (kx, ky)\n对应一个\n以 θ=arcsin(λkr/n)\n射向焦点的\n平面波 (plane wave)',
    va='center', ha='left', fontsize=7.8, transform=axes[2].transData,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF7ED', edgecolor='#F59E0B', lw=1))

plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig(f'{OUT}/optics_fig4_pupil_function.png', dpi=160,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 4 done.')
print('\nAll figures saved to', OUT)

# =============================================================================
# 图 5：主面 (Principal Planes) H 与 H' 详细讲解
#   Panel A: 定义 — 单位横向放大率
#   Panel B: 几何求法 — 延伸光线交点
#   Panel C: 三种典型情形对比
# =============================================================================

fig5, axes5 = plt.subplots(1, 3, figsize=(18, 7))
fig5.patch.set_facecolor('white')

BOX2 = dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.90)

# ── Panel A: 单位放大率定义 ───────────────────────────────────────────────
axA = axes5[0]
axA.set_xlim(-5.5, 5.5)
axA.set_ylim(-2.8, 3.6)
axA.set_aspect('equal')
axA.axis('off')
axA.set_title('(A) 定义: 单位横向放大率\n(Definition: Unit Transverse Magnification)',
              fontsize=10, fontweight='bold', pad=6)

# 光轴
axA.annotate('', xy=(5.2, 0), xytext=(-5.2, 0),
             arrowprops=dict(arrowstyle='->', color='#888', lw=1.0, mutation_scale=9))

# 黑盒光学系统 (black-box optical system)
bbox_x0, bbox_x1, bbox_h = -1.2, 1.2, 1.8
rect_bb = mpatches.FancyBboxPatch(
    (bbox_x0, -bbox_h), bbox_x1-bbox_x0, 2*bbox_h,
    boxstyle="round,pad=0.1",
    facecolor='#F3F4F6', edgecolor='#374151', lw=2, zorder=3, alpha=0.85)
axA.add_patch(rect_bb)
axA.text(0, 0, '任意光学系统\n(Optical System\n  — Black Box)',
         ha='center', va='center', fontsize=8, color='#374151', zorder=4)

# 主面 H 和 H' 竖线
Hxa, Hpxa = -0.5, 0.5
vline(axA, Hxa,  -2.0, 3.2, color='#7C3AED', lw=2.2, ls='-.')
vline(axA, Hpxa, -2.0, 3.2, color='#DB2777', lw=2.2, ls='-.')
axA.text(Hxa,  3.25, '前主面 H\n(Front Principal\nPlane, FPP)',
         ha='center', va='bottom', fontsize=8.5, color='#5B21B6', bbox=BOX2)
axA.text(Hpxa, 3.25, '后主面 H\'\n(Back Principal\nPlane, RPP)',
         ha='center', va='bottom', fontsize=8.5, color='#9D174D', bbox=BOX2)

# 物点 Q 在 H 上，像点 Q' 在 H' 上，等高 (unit magnification)
Qy = 1.1
axA.plot(Hxa,  Qy, 's', ms=10, color='#3B82F6', zorder=6)
axA.plot(Hpxa, Qy, 's', ms=10, color='#EF4444', zorder=6)

axA.annotate('', xy=(Hxa, Qy), xytext=(Hxa, 0),
             arrowprops=dict(arrowstyle='->', color='#3B82F6', lw=1.8, mutation_scale=11))
axA.annotate('', xy=(Hpxa, 0), xytext=(Hpxa, Qy),
             arrowprops=dict(arrowstyle='->', color='#EF4444', lw=1.8, mutation_scale=11))

axA.text(Hxa-0.18, Qy/2, 'h', ha='right', va='center', fontsize=12,
         color='#3B82F6', fontweight='bold')
axA.text(Hpxa+0.18, Qy/2, 'h', ha='left', va='center', fontsize=12,
         color='#EF4444', fontweight='bold')

axA.text(Hxa, Qy+0.18, 'Q (物点)', ha='center', va='bottom', fontsize=8,
         color='#1D4ED8', bbox=BOX2)
axA.text(Hpxa, Qy+0.18, "Q' (像点)", ha='center', va='bottom', fontsize=8,
         color='#991B1B', bbox=BOX2)

# 放大率标注
axA.text(0, -1.6,
         '主面定义：Q → Q\' 横向放大率 m = +1\n'
         '(物体在 H 面，像在 H\' 面，等大正立)\n'
         'Principal planes: magnification m = +1\n'
         '(Object at H imaged to H\' at same height)',
         ha='center', va='center', fontsize=8,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#EDE9FE',
                   edgecolor='#7C3AED', lw=1.2))

# ── Panel B: 几何求法 ─────────────────────────────────────────────────────
axB = axes5[1]
axB.set_xlim(-5.5, 5.5)
axB.set_ylim(-2.8, 3.6)
axB.set_aspect('equal')
axB.axis('off')
axB.set_title('(B) 几何求法: 延伸光线求交点\n(Geometric Method: Extend Rays to Find H/H\')',
              fontsize=10, fontweight='bold', pad=6)

# 光轴
axB.annotate('', xy=(5.2, 0), xytext=(-5.2, 0),
             arrowprops=dict(arrowstyle='->', color='#888', lw=1.0, mutation_scale=9))

# 黑盒
rectB = mpatches.FancyBboxPatch(
    (-1.0, -1.5), 2.0, 3.0,
    boxstyle="round,pad=0.1",
    facecolor='#F3F4F6', edgecolor='#374151', lw=2, zorder=3, alpha=0.85)
axB.add_patch(rectB)
axB.text(0, 0, '光学系统\n(Optical System)',
         ha='center', va='center', fontsize=8, color='#374151', zorder=4)

# 主面 H H'
Hxb, Hpxb = -0.4, 0.6
vline(axB, Hxb,  -1.8, 3.0, color='#7C3AED', lw=2.0, ls='-.')
vline(axB, Hpxb, -1.8, 3.0, color='#DB2777', lw=2.0, ls='-.')
axB.text(Hxb,  3.05, 'H', ha='center', va='bottom', fontsize=11,
         color='#5B21B6', fontweight='bold')
axB.text(Hpxb, 3.05, "H'", ha='center', va='bottom', fontsize=11,
         color='#9D174D', fontweight='bold')

# 后焦点 F'
Fpxb = Hpxb + 3.0
axB.plot(Fpxb, 0, 'o', ms=7, color='#059669', zorder=6)
axB.text(Fpxb+0.12, -0.25, "F'", ha='left', va='top', fontsize=10,
         color='#065F46', fontweight='bold')
vline(axB, Fpxb, -0.4, 0.4, color='#059669', lw=1.5, ls='-')

# 光线 1：平行入射（高度 y=1）
yb = 1.0
# 入射段（平行）
axB.plot([-5.0, -1.0], [yb, yb], color='#EF4444', lw=2.0)
arrow(axB, -5.0, yb, -3.0, yb, color='#EF4444', lw=2.0, ms=11)
# 出射段（过 F'，从 H' 处射出）
m_out = (0 - yb) / (Fpxb - Hpxb)
axB.plot([Hpxb, 5.0], [yb, yb + m_out*(5.0-Hpxb)], color='#EF4444', lw=2.0)
arrow(axB, Hpxb, yb, Fpxb+0.5, yb+m_out*0.5, color='#EF4444', lw=2.0, ms=11)

# 延伸入射光线（虚线，向右延伸）
axB.plot([Hpxb, 5.0], [yb, yb], color='#EF4444', lw=1.2, ls='--', alpha=0.55)
# 延伸出射光线（虚线，向左延伸回 H' 以左）
axB.plot([-5.0, Hpxb], [yb+m_out*(-5.0-Hpxb), yb], color='#EF4444', lw=1.2, ls='--', alpha=0.55)

# 交点标记（两延长线在 H' 处以高度 yb 相交）
axB.plot(Hpxb, yb, 'D', ms=9, color='#DC2626', zorder=7)
axB.annotate('两延伸线在 H\' 处交于同一高度\n(Extended rays meet at H\' at same height)',
             xy=(Hpxb, yb), xytext=(Hpxb+0.5, yb+0.9),
             fontsize=7.5, color='#991B1B',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEE2E2', edgecolor='#EF4444', lw=1),
             arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.2))

# 焦距双箭头
axB.annotate('', xy=(Fpxb, -1.5), xytext=(Hpxb, -1.5),
             arrowprops=dict(arrowstyle='<->', color='#065F46', lw=1.3, mutation_scale=10))
axB.text((Hpxb+Fpxb)/2, -1.72, "后焦距 f'\n(Back Focal Length, EFL)\n从 H' 量到 F'",
         ha='center', va='top', fontsize=7.8, color='#065F46', bbox=BOX2)

# 说明
axB.text(0, -2.55,
         '求 H\' 的方法: 平行入射光 → 出射光延伸\n'
         '与平行入射光延伸线的交点 → 即 H\' 面\n'
         'To find H\': extend parallel input & output rays;\n'
         'intersection defines back principal plane H\'',
         ha='center', va='center', fontsize=7.8,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FCE7F3',
                   edgecolor='#DB2777', lw=1.2))

# ── Panel C: 三种典型情形对比 ─────────────────────────────────────────────
axC = axes5[2]
axC.set_xlim(0, 10)
axC.set_ylim(-1.0, 10.5)
axC.set_aspect('equal')
axC.axis('off')
axC.set_title('(C) 三种典型情形对比\n(Three Typical Cases)',
              fontsize=10, fontweight='bold', pad=6)

def draw_case(ax, y_center, label_left, lens_x0, lens_x1, h_frac,
              Hx_rel, Hpx_rel, Fx_abs, Fpx_abs,
              flen, color_H='#7C3AED', color_Hp='#DB2777',
              note=''):
    """在水平带 y_center 处画一个对比案例"""
    yw = 1.5   # 半宽
    lw_ax = 0.9
    # 光轴
    ax.annotate('', xy=(9.5, y_center), xytext=(0.2, y_center),
                arrowprops=dict(arrowstyle='->', color='#888', lw=lw_ax, mutation_scale=7))
    # 透镜/系统框
    rect = mpatches.FancyBboxPatch(
        (lens_x0, y_center-h_frac), lens_x1-lens_x0, 2*h_frac,
        boxstyle="round,pad=0.05",
        facecolor='#BFDBFE', edgecolor='#3B82F6', lw=1.5, zorder=3, alpha=0.7)
    ax.add_patch(rect)
    # H 和 H'
    Habs  = lens_x0 + (lens_x1-lens_x0)*Hx_rel
    Hpabs = lens_x0 + (lens_x1-lens_x0)*Hpx_rel
    for xv, col in [(Habs, color_H), (Hpabs, color_Hp)]:
        ax.plot([xv, xv], [y_center-yw, y_center+yw], color=col, lw=2.0, ls='-.', zorder=4)
    # 焦点
    ax.plot(Fx_abs,  y_center, 'o', ms=5, color='#D97706', zorder=6)
    ax.plot(Fpx_abs, y_center, 'o', ms=5, color='#059669', zorder=6)
    # 标注
    ax.text(Habs,  y_center+yw+0.08, 'H', ha='center', va='bottom', fontsize=8,
            color=color_H, fontweight='bold')
    ax.text(Hpabs, y_center+yw+0.08, "H'", ha='center', va='bottom', fontsize=8,
            color=color_Hp, fontweight='bold')
    ax.text(Fx_abs,  y_center-0.2, 'F', ha='center', va='top', fontsize=7.5, color='#92400E')
    ax.text(Fpx_abs, y_center-0.2, "F'", ha='center', va='top', fontsize=7.5, color='#065F46')
    # 情形名称
    ax.text(0.35, y_center, label_left, ha='left', va='center', fontsize=8.2,
            color='#1F2937', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F9FAFB', edgecolor='#D1D5DB', lw=0.8))
    # 备注
    if note:
        ax.text(5.0, y_center-yw-0.18, note, ha='center', va='top', fontsize=7.2,
                color='#4B5563', style='italic')

# 情形 1：薄透镜 (H = H' 重合，在透镜中心)
draw_case(axC, y_center=8.5,
          label_left='① 薄透镜\n(Thin Lens)',
          lens_x0=4.7, lens_x1=5.3, h_frac=1.1,
          Hx_rel=0.5, Hpx_rel=0.5,   # 重合
          Fx_abs=2.5, Fpx_abs=7.5,
          flen=2.5,
          note='H = H\' (coincide at lens center, f = f\')')

# 情形 2：厚玻璃透镜 (H 和 H' 在玻璃内分离)
draw_case(axC, y_center=5.2,
          label_left='② 厚透镜\n(Thick Lens)',
          lens_x0=3.5, lens_x1=6.5, h_frac=1.1,
          Hx_rel=0.3, Hpx_rel=0.7,   # 在玻璃内分离
          Fx_abs=1.2, Fpx_abs=8.8,
          flen=2.5,
          note='H and H\' inside glass, separated (f ~ f\')')

# 情形 3：显微物镜 (H' 在物镜外部，焦距很短)
draw_case(axC, y_center=1.8,
          label_left='③ 显微物镜\n(Micro-Objective)',
          lens_x0=5.0, lens_x1=8.5, h_frac=1.1,
          Hx_rel=-0.8, Hpx_rel=1.4,  # H 在透镜前，H' 在透镜后外部
          Fx_abs=4.1, Fpx_abs=9.5,
          flen=1.5,
          note='H or H\' may lie outside physical lens body')

# 图例
axC.plot([], [], color='#7C3AED', lw=2, ls='-.',
         label='前主面 H (Front Principal Plane)')
axC.plot([], [], color='#DB2777', lw=2, ls='-.',
         label='后主面 H\' (Back Principal Plane)')
axC.plot([], [], 'o', ms=5, color='#D97706', label='前焦点 F (Front Focal Point)')
axC.plot([], [], 'o', ms=5, color='#059669', label='后焦点 F\' (Back Focal Point)')
axC.legend(loc='lower center', fontsize=7.5, framealpha=0.9, edgecolor='#D1D5DB',
           bbox_to_anchor=(0.5, -0.01), ncol=2)

fig5.suptitle(
    '图5  主面 (Principal Planes) H 与 H\' 详解\n'
    'Principal Planes: Definition, Geometric Construction, and Typical Cases',
    fontsize=12, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0.02, 1, 0.97])
plt.savefig(f'{OUT}/optics_fig5_principal_planes.png', dpi=160,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 5 (principal planes) done.')

# =============================================================================
# 图 6：荧光显微镜激发端光路
#   上半：检测端（样品 → 物镜 → BFP → 管镜 → 相机）
#   下半：激发端（激光 → SLM/BFP → relay → 物镜 → 样品）
#   + 标注 SLM 可放位置 vs 必须在共轭瞳面
# =============================================================================

fig6, (axD, axE) = plt.subplots(2, 1, figsize=(17, 10))
fig6.patch.set_facecolor('white')

# ─── 通用辅助 ──────────────────────────────────────────
def draw_lens6(ax, x, h=1.4, color='#3B82F6', lw=2.8):
    ax.plot([x, x], [-h, h], color=color, lw=lw, solid_capstyle='round', zorder=4)
    ax.annotate('', xy=(x,  h+0.18), xytext=(x,  h),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=13), zorder=4)
    ax.annotate('', xy=(x, -h-0.18), xytext=(x, -h),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=13), zorder=4)

def slm_patch(ax, x, h=1.2, color='#059669', label='SLM'):
    rect = mpatches.FancyBboxPatch((x-0.18, -h), 0.36, 2*h,
        boxstyle='round,pad=0.05', facecolor='#D1FAE5', edgecolor=color, lw=2, zorder=4)
    ax.add_patch(rect)
    ax.text(x, h+0.25, label, ha='center', va='bottom', fontsize=8,
            color=color, fontweight='bold')

def vline6(ax, x, y0, y1, color='#aaa', lw=1.3, ls='--', zorder=2):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, ls=ls, zorder=zorder)

BOX6 = dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='none', alpha=0.9)

# ══════════════════════════════════════════════════════
# 上图：检测端 Detection Path
# ══════════════════════════════════════════════════════
ax = axD
ax.set_xlim(-0.5, 17)
ax.set_ylim(-2.5, 3.2)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('检测端 (Detection Path)  — 样品发出荧光 → 相机成像',
             fontsize=11, fontweight='bold', pad=5, color='#1E3A8A')

# 光轴 (右向)
ax.annotate('', xy=(16.8, 0), xytext=(-0.3, 0),
            arrowprops=dict(arrowstyle='->', color='#888', lw=1.0, mutation_scale=9))

# 位置参数
xS  = 1.0   # 样品
xO  = 3.5   # 物镜
xB  = 5.5   # BFP / BPP
xTL = 11.5  # 管镜
xCam= 14.0  # 相机

# 竖线
vline6(ax, xS,  -1.8, 2.8, color='#D97706', lw=2.0)
vline6(ax, xB,  -1.8, 2.8, color='#059669', lw=2.0)
vline6(ax, xCam,-1.8, 2.8, color='#DC2626', lw=2.0)

# 物镜、管镜
draw_lens6(ax, xO,  h=1.5, color='#3B82F6')
draw_lens6(ax, xTL, h=1.3, color='#7C3AED')

# 样品点光源
ax.plot(xS, 0.0, '*', ms=16, color='gold', markeredgecolor='#92400E', zorder=7)
ax.plot(xS, 0.7, 'o', ms=7,  color='#EF4444', zorder=7)

# 轴上点 → 平行光 (3束)
ys = [-1.1, 0.0, 1.1]
cs = ['#60A5FA', '#3B82F6', '#1D4ED8']
for yp, cp in zip(ys, cs):
    ax.plot([xO, xTL], [yp, yp], color=cp, lw=1.8)
    arrow(ax, xO+(xTL-xO)*0.3, yp, xO+(xTL-xO)*0.6, yp, color=cp, lw=1.8, ms=9)
    ax.plot([xTL, xCam], [yp, 0], color=cp, lw=1.8)

# 标注
ax.text(xS,  2.9, '样品面\n(Sample / FFP)', ha='center', va='bottom', fontsize=8, color='#92400E', bbox=BOX6)
ax.text(xO,  2.9, '物镜\n(Objective)', ha='center', va='bottom', fontsize=8.5, color='#1E3A8A', bbox=BOX6)
ax.text(xB,  2.9, '后瞳面\n(BFP / BPP)\n← SLM 检测端滤波', ha='center', va='bottom', fontsize=8, color='#065F46', bbox=BOX6)
ax.text(xTL, 2.9, '管镜\n(Tube Lens)', ha='center', va='bottom', fontsize=8.5, color='#5B21B6', bbox=BOX6)
ax.text(xCam,2.9, '相机\n(Camera)', ha='center', va='bottom', fontsize=8, color='#991B1B', bbox=BOX6)

# 平行光区
mid = (xO + xTL) / 2
ax.text(mid, 1.55, '平行光区 (Infinity Space)\nSLM / 滤光片可放此区间任意位置（原因见说明）',
        ha='center', va='bottom', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#F0FDF4', edgecolor='#6EE7B7', lw=1.2))
ax.annotate('', xy=(xTL-0.15, 1.45), xytext=(xO+0.15, 1.45),
            arrowprops=dict(arrowstyle='<->', color='#10B981', lw=1.3, mutation_scale=10))

ax.text(0.0, -2.2, '方向：荧光从样品出发 →', ha='left', va='center',
        fontsize=8.5, color='#374151', style='italic')

# ══════════════════════════════════════════════════════
# 下图：激发端 Excitation Path
# ══════════════════════════════════════════════════════
ax2 = axE
ax2.set_xlim(-0.5, 17)
ax2.set_ylim(-2.5, 3.2)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('激发端 (Excitation Path)  — 激光经 SLM 整形后聚焦到样品',
              fontsize=11, fontweight='bold', pad=5, color='#7C2D12')

# 光轴（左向，激光从右射向左边样品）
ax2.annotate('', xy=(-0.3, 0), xytext=(16.8, 0),
             arrowprops=dict(arrowstyle='->', color='#888', lw=1.0, mutation_scale=9))

# 位置参数（激光从右到左传播，坐标与上图镜像对应）
xLaser = 15.5  # 激光源
xSLM   = 12.5  # SLM（位于物镜后瞳面的共轭面）
xRL1   = 10.5  # Relay Lens 1
xRL2   =  8.5  # Relay Lens 2  → 两者之间为SLM的共轭瞳面位置
xObj2  =  3.5  # 物镜（激发端）
xSamp2 =  1.0  # 样品

vline6(ax2, xSLM,  -1.8, 2.8, color='#059669', lw=2.0)
vline6(ax2, xSamp2,-1.8, 2.8, color='#D97706', lw=2.0)

# 激光源
ax2.plot(xLaser, 0, '>', ms=12, color='#DC2626', zorder=7)
ax2.text(xLaser+0.15, 0.3, '激光\n(Laser)', ha='left', va='bottom', fontsize=8.5,
         color='#DC2626', fontweight='bold')

# SLM
slm_patch(ax2, xSLM, h=1.3, color='#059669', label='SLM\n(后瞳共轭面)')

# Relay lenses
draw_lens6(ax2, xRL1, h=1.2, color='#F59E0B')
draw_lens6(ax2, xRL2, h=1.2, color='#F59E0B')
draw_lens6(ax2, xObj2, h=1.5, color='#3B82F6')

# 激光光束（平行光从激光→SLM）
for yp in [-0.9, 0.0, 0.9]:
    cp = '#EF4444'
    ax2.plot([xLaser-0.1, xSLM+0.2], [yp, yp], color=cp, lw=1.7, alpha=0.8)
    arrow(ax2, xLaser-0.5, yp, xLaser-2.5, yp, color=cp, lw=1.7, ms=9)

# SLM后 → Relay → 物镜（整形后的光场）
lls_ys = [-0.8, -0.3, 0.3, 0.8]
c_lls = '#8B5CF6'
for yp in lls_ys:
    ax2.plot([xSLM-0.2, xRL1], [yp*1.2, yp], color=c_lls, lw=1.5, alpha=0.8)
    ax2.plot([xRL1, xRL2],     [yp, yp],      color=c_lls, lw=1.5, alpha=0.8)
    ax2.plot([xRL2, xObj2],    [yp, yp*1.1],  color=c_lls, lw=1.5, alpha=0.8)
    ax2.plot([xObj2, xSamp2],  [yp*1.1, 0],   color=c_lls, lw=1.5, alpha=0.8)

# 样品焦点
ax2.plot(xSamp2, 0, '*', ms=18, color='gold', markeredgecolor='#92400E', zorder=7)

# 标注
ax2.text(xSLM,   2.9, 'SLM 位于后瞳共轭面\n(Conjugate of BFP)\n精确控制瞳函数 P(kx,ky)',
         ha='center', va='bottom', fontsize=8, color='#065F46', bbox=BOX6)
ax2.text(xRL1+0.0, -2.2, 'Relay Lens 对 (4f system)\n将 SLM 面成像到物镜后瞳面',
         ha='center', va='center', fontsize=8, color='#92400E',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEF3C7', edgecolor='#F59E0B', lw=1))
ax2.text(xRL1, 2.9, 'Relay L1', ha='center', va='bottom', fontsize=8, color='#B45309', bbox=BOX6)
ax2.text(xRL2, 2.9, 'Relay L2', ha='center', va='bottom', fontsize=8, color='#B45309', bbox=BOX6)
ax2.text(xObj2, 2.9, '物镜\n(Objective)', ha='center', va='bottom', fontsize=8.5, color='#1E3A8A', bbox=BOX6)
ax2.text(xSamp2, 2.9, '样品\n(Sample / FFP)', ha='center', va='bottom', fontsize=8, color='#92400E', bbox=BOX6)

# SLM偏置说明
ax2.annotate(
    'Q: SLM 能不能不放在后瞳共轭面？\n'
    '→ 在平行光区任意位置：光束仍是平行的，\n'
    '   但 SLM 像素不再对应固定的 kx/ky 角度，\n'
    '   会产生"串扰"（不同角度混叠）。\n'
    '→ 偏离越大，串扰越严重；实验中通常\n'
    '   用 4f relay 把 SLM 精确成像到物镜瞳面。',
    xy=(xSLM, -0.5), xytext=(xSLM+2.5, -1.8),
    fontsize=7.8, color='#1F2937',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#EFF6FF', edgecolor='#93C5FD', lw=1.3),
    arrowprops=dict(arrowstyle='->', color='#3B82F6', lw=1.2))

ax2.text(16.5, -2.2, '← 方向：激光从右到左传播', ha='right', va='center',
         fontsize=8.5, color='#374151', style='italic')

fig6.suptitle(
    '图6  荧光显微镜两条光路对比\n'
    'Fluorescence Microscope: Detection vs Excitation Path',
    fontsize=12, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{OUT}/optics_fig6_fluorescence_paths.png', dpi=160,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 6 (fluorescence paths) done.')


