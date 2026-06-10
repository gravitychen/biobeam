# 03: 基于 cell11_Settings.txt 模拟光片和 effective PSF

用途：读取 `cell11_Settings.txt` 中的实验参数，生成 LLSPY/SLM 光片入口场，并继续计算 dither 后的 excitation light sheet、BioBeam detection PSF 和 effective PSF。

运行：

```powershell
cd "D:\codes\biobeam\LLSPY 光片导入进 BioBeam\03_基于_cell11_Settings_模拟光片和_effective_PSF"
python -m marimo edit "cell11_settings_to_biobeam_effective_psf__RUN_marimo_edit.py"
```

当前直接映射的 cell11 参数：

- excitation wavelength: `488 nm`
- annular mask in settings: `innerNA = 0.50`, `outerNA = 0.55`
- simulation excitation annulus default: innerNA `0.50`, outerNA `0.55`
- simulation excitation objective NA default: `0.80`; this is the objective aperture limit, not annular outerNA
- objective NA in settings: `0.80`, ambiguous; not proven to be detection NA
- detection NA control defaults to `0.90`
- X galvo dither metadata: `51 px`, `0.1 um` step
- DOE beam metadata: `DOE installed? = FALSE`, `# of beams = 10`, `Beam spacing = 3 um`
- sample stack acquisition metadata: `201 planes`, `0.4 um` step; this is the sample piezo stack acquisition range, not the PSF propagation window
- detection magnification、ROI、stage angle 作为 metadata 显示

说明：

- `Magnification from SLM to back pupil = 0.5` 不等同于当前 helper 里的 LLSPY `mag`，所以 notebook 只显示它，不直接套入 `FillChipParams.mag`。
- `mag/crop/fudge/fillchip` 仍作为控件保留，因为 `cell11_Settings.txt` 没有完整保存这些 LLSPY SLM 图案生成参数。
- `Angle between stage and bessel beam = 31.1089 deg` 当前不用于 PSF 坐标变换；notebook 里 stage angle 按 `0 deg` 处理。
- `# of beams = 10` 和 `Beam spacing = 3 um` 在 DOE 设置段里，但 `DOE installed? = FALSE`，所以默认不硬套。notebook 里提供 `Use explicit # beams / spacing from settings` 开关，确认需要时再启用。
- detection PSF 默认优先使用 BioBeam `focus_field_beam`；如果本机 OpenCL/BioBeam 不可用，会自动退回 Gaussian fallback。
