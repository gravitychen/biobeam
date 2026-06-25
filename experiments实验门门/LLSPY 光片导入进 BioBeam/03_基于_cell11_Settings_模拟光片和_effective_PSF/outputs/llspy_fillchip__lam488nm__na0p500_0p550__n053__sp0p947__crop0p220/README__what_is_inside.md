# LLSPY 光片导入 BioBeam 的中间结果

- `slm_binary_mask`: LLSPY 的 binary SLM mask。
- `pupil_plane_intensity`: 经过 annular mask 后的 pupil plane intensity。
- `sample_plane_intensity`: LLSPY Fourier 链路得到的 sample plane intensity。
- `illumination_volume_cpu`: dither 后并转换到 detection objective 坐标系的 illumination volume。
- `detection_psf_cpu`: detection PSF；优先来自 BioBeam，失败时退回高斯近似。
- `detection_psf_gaussian_cpu`: 兼容旧脚本的别名，内容等同于 `detection_psf_cpu`。
- `effective_psf_cpu`: `illumination_volume_cpu * detection_psf_cpu`。
- 坐标系: `(z_detection, y_excitation_propagation, x)`。