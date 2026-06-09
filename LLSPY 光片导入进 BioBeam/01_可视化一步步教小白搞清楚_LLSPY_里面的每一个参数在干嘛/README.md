# 01: 渐进式 LLSPY 参数教学

用途：用 marimo 从 1 个参数开始教 LLSPY `Square Lattice, Fill Chip`，逐步增加到 10 个参数。

运行：

```powershell
cd "C:\Code\biobeam\LLSPY 光片导入进 BioBeam\01_可视化一步步教小白搞清楚_LLSPY_里面的每一个参数在干嘛"
python -m marimo edit "llspy_parameters_tutorial_marimo__IN_llspy_gui_defaults__OUT_interactive_parameter_figures__RUN_marimo_edit.py"
```

教学方式：

1. 阶段 1 只显示 `Wavelength`。
2. 阶段 2 显示 `Wavelength + Inner NA`。
3. 阶段 3 显示前三个参数。
4. 直到阶段 10 才显示完整 10 个参数。

没显示的参数会锁定在 LLSPY Fill Chip 默认值上，这样每一阶段只需要理解新加入的那个参数。
