# 04: cell11 parameters in original BioBeam PSF workflow

This notebook does not use the 03 LLSPY/SLM bridge.

It keeps the original BioBeam script workflow:

`focus_field_lattice -> dither -> focus_field_beam -> np.swapaxes(excitation, 0, 1) -> excitation * detection`

Inputs:

- original BioBeam script: `d:\codes\biobeam\bessel_lattice_lightsheet_psf_visualization_plotly可视化LLSM激发检测光.py`
- cell11 settings: `d:\codes\microscopy_proj\microscopy_data\New_lattice_sheet_data\20220329-30_CD63_deskew[433GB]\cell11\cell11_Settings.txt`

References compared against the generated PSF:

- old BioBeam PSF: `d:\codes\biobeam\psfdata\bessel_lattice_lightsheet_effective_psf.tif`
- sample PSF488: `d:\codes\microscopy_proj\microscopy_data\Lattice_light_sheet_microscope_sample_data\PSF\PSF488.tif`

Run:

```powershell
cd "D:\codes\biobeam\LLSPY 光片导入进 BioBeam\04_比较_cell11套用原始BioBeam脚本_和_旧PSF"
python -m marimo edit .\cell11_params_in_original_biobeam_compare_old_psf__RUN_marimo_edit.py
```

Default parameters:

- excitation wavelength from cell11 laser, default 488 nm
- NA1/NA2 from cell11 `innerNA/outerNA`, default 0.50/0.55
- excitation objective NA is used as an upper cap for outer NA, default 0.80
- detection NA default 0.90
- detection wavelength default 525 nm
- `kpoints` keeps the original BioBeam script default 6; the notebook also has a switch to try cell11 `# of beams=10` as `kpoints`

