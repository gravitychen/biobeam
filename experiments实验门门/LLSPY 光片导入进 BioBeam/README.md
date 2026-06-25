# LLSPY 光片导入进 BioBeam

这个文件夹把任务拆成两步：

1. `01_可视化一步步教小白搞清楚_LLSPY_里面的每一个参数在干嘛`

   目标是先看懂 LLSPY GUI 参数。这个 notebook 会交互显示 `binary SLM mask`、`Intensity After Mask`、`Intensity At Sample`，并解释每个参数改变了哪一步。

2. `02_真正把_LLSPY_导入进_BioBeam`

   目标是把 LLSPY 生成的光片结果整理成后续 BioBeam / Effective PSF 可以使用的数据流。当前机器没有 NVIDIA GPU/OpenCL 时，会自动用 CPU 角谱传播和高斯 detection PSF fallback 先跑通。

共享代码：

- `llspy_fillchip_shared_math__IN_llspy_gui_params__OUT_slm_pupil_sample_arrays.py`

运行 01：

```powershell
cd "C:\Code\biobeam\LLSPY 光片导入进 BioBeam\01_可视化一步步教小白搞清楚_LLSPY_里面的每一个参数在干嘛"
python -m marimo edit "llspy_parameters_tutorial_marimo__IN_llspy_gui_defaults__OUT_interactive_parameter_figures__RUN_marimo_edit.py"
```

运行 02：

```powershell
cd "C:\Code\biobeam\LLSPY 光片导入进 BioBeam\02_真正把_LLSPY_导入进_BioBeam"
python -m marimo edit "llspy_light_sheet_to_biobeam_marimo__IN_llspy_slm_pattern__OUT_biobeam_effective_psf_inputs__RUN_marimo_edit.py"
```

如果当前 PowerShell 里的 `python` 不是 conda base，可以直接指定：

```powershell
C:\Application\Miniconda\python.exe -m marimo edit "脚本名.py"
```
