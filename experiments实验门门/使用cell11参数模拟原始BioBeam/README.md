# 使用 cell11 参数模拟原始 BioBeam LLSM PSF

这个目录里的 marimo notebook 用于在原始 BioBeam LLSM PSF 工作流上套用 cell11 的采集参数，并把新模拟 PSF 与旧模拟 PSF、真实测量 PSF 进行比较。

核心流程保持为：

```text
focus_field_lattice -> dither -> focus_field_beam -> np.swapaxes(excitation, 0, 1) -> excitation * detection
```

## 当前生成的 PSF

| 项目 | 路径 | 说明 |
|---|---|---|
| 最新 cell11 参数模拟 PSF | `C:\Code\biobeam\psfdata\cell11_current_biobeam_effective_psf.tif` | 当前 notebook 每次重新模拟后会覆盖保存到这里 |
| 旧 BioBeam 模拟 PSF | `C:\Code\biobeam\psfdata\bessel_lattice_lightsheet_effective_psf.tif` | 旧参数模拟结果，用作历史参考 |
| 真实测量 LLSM PSF | `C:\Code\biobeam\psfdata\20220329_488_square_0p55-0p50.tif` | 实际采集的 488 nm square lattice PSF |
| 对比可视化脚本 | `C:\Code\biobeam\psfdata\compare_psf_slices.py` | 横向比较 psfdata 里的所有 PSF |

## 当前 cell11 模拟参数

| 参数 | 当前值 | 来源 / 含义 |
|---|---:|---|
| 输出 shape | `(52, 52, 52)` | 默认与旧 BioBeam PSF 的 shape 一致 |
| voxel size | `0.10 um` | 当前模拟网格的各向同性体素尺寸 |
| excitation wavelength | `488 nm` | 来自 cell11 采集激发激光 |
| annular inner NA / `NA1` | `0.50` | 来自 cell11 `[Annular Mask] innerNA` |
| annular outer NA / `NA2` | `0.55` | 来自 cell11 `[Annular Mask] outerNA` |
| excitation objective NA cap | `0.80` | 来自 cell11 `[Objective] Numerical Aperature`，用于限制 outer NA 上限 |
| lattice geometry | `square lattice` | 文件名中的 `square` 对应 BioBeam `kpoints=4` |
| BioBeam `kpoints` | `4` | `4` 表示 square lattice；旧脚本曾使用 `6`，更接近 hex lattice |
| sigma / sigma phi | `0.10` | pupil 上 lattice points 的 Gaussian smearing 参数；会影响光片形态 |
| detection wavelength | `525 nm` | 当前模拟默认 emission wavelength |
| detection NA | `1.10` | 当前按典型 LLSM water-dipping detection objective 设置 |
| medium refractive index / `n0` | `1.33` | 水环境近似折射率 |
| dither excitation | `True` | 对 lattice excitation pattern 做扫描平均，得到 lightsheet |
| dither step rule | `auto` | 使用原始脚本规则自动计算 dither step |
| dither step pixels | `2 px` | 当前默认参数下自动得到的扫描步长 |
| scan positions | `26` | shape 为 52、step 为 2 时覆盖整个扫描方向 |

## 与 cell11 采集文件相关的参数

| cell11 文件字段 | 数值 | 在当前模拟中的使用方式 |
|---|---:|---|
| `innerNA` | `0.50` | 用作 BioBeam `NA1` |
| `outerNA` | `0.55` | 用作 BioBeam `NA2` |
| `Numerical Aperature` | `0.80` | 用作 excitation objective NA cap |
| `Excitation Filter, Laser...` | `488 nm` | 用作 excitation wavelength |
| `DOE installed?` | `FALSE` | 记录但不直接改变当前默认模拟 |
| `# of beams` | `10` | notebook 中保留开关，可临时作为 `kpoints` 试算；默认不用 |
| `Beam spacing (um)` | `3.0 um` | 记录但不直接映射到 BioBeam `kpoints` |
| `Angle between stage and bessel beam` | `31.1089 deg` | 记录采集/deskew 角度；当前 BioBeam forward simulation 不直接使用 |

## 参数解释

| 参数 | 解释 |
|---|---|
| Inner NA / Outer NA | 定义 excitation pupil 上的环形孔径范围。当前 cell11 使用 `0.50-0.55`，比旧 BioBeam 默认的 `0.44-0.58` 更贴近真实采集文件名 `0p55-0p50`。 |
| `square` | 指 lattice pattern 的几何形状。在 BioBeam 中用 `kpoints=4` 表示 square lattice。 |
| `sigma` / `sigma phi` | 表示 pupil 上每个 lattice point 的 Gaussian 展宽。它不是真实空间里的 PSF 宽度，而是影响 pupil 点扩散和最终 light sheet 形态的模拟参数。 |
| `dither step pixels` | 表示把 lattice excitation pattern 沿扫描方向平移并平均时的像素步长。步长越小，扫描平均越密。 |
| `detection NA` | detection objective 的 NA。当前设置为 `1.10`，更接近常见 LLSM high-NA water-dipping detection objective。 |

## 运行方式

```powershell
cd "C:\Code\biobeam\experiments实验门门\使用cell11参数模拟原始BioBeam"
python -m marimo edit .\cell11_params_in_original_biobeam_compare_old_psf__RUN_marimo_edit.py
```

运行 notebook 后，当前模拟 PSF 会保存到：

```text
C:\Code\biobeam\psfdata\cell11_current_biobeam_effective_psf.tif
```

如需重新生成所有 PSF 的横向切片比较图：

```powershell
python C:\Code\biobeam\psfdata\compare_psf_slices.py
```

输出图：

```text
C:\Code\biobeam\psfdata\psf_slices_comparison.png
```
