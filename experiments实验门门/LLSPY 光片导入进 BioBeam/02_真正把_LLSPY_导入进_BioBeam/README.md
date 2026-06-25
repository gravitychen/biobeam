# 02: 渐进式把 LLSPY 光片导入 BioBeam

用途：用 marimo 按物理原理逐步叠加 LLSPY 到 BioBeam / Effective PSF 的完整链路。

运行：

```powershell
cd "C:\Code\biobeam\LLSPY 光片导入进 BioBeam\02_真正把_LLSPY_导入进_BioBeam"
python -m marimo edit "llspy_light_sheet_to_biobeam_marimo__IN_llspy_slm_pattern__OUT_biobeam_effective_psf_inputs__RUN_marimo_edit.py"
```

教学阶段：

1. 参数解析：GUI 参数变成 spacing、# Beams、pupil ring 半径。
2. SLM 二值相位图：生成真正写入 SLM 的 0/pi mask。
3. Pupil plane + annular mask：看傅立叶变换后如何被 Inner/Outer NA 筛选。
4. Sample plane 光片：看 LLSPY 光片在样品平面的强度。
5. 3D illumination volume：把 complex pupil field 沿 z 传播。
6. Detection PSF：加入检测通道 PSF。当前是 CPU 高斯 fallback。
7. Effective PSF：`effective_psf = illumination_volume * detection_psf`，并允许保存输出。

说明：

- 当前机器没有 NVIDIA/OpenCL 也能跑，因为第 5 阶段使用 CPU 角谱传播。
- 后续如果 BioBeam GPU/OpenCL 可用，第 5 阶段可以替换成 BioBeam BPM propagation。
