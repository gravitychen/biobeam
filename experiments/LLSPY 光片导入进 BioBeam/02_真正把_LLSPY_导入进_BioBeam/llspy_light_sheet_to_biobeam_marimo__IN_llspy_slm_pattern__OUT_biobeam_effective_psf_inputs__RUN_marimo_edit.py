"""
用途: 渐进式解释并实现 LLSPY 光片导入 BioBeam/Effective PSF 的完整数据流。
输入: LLSPY GUI 参数；原理阶段 slider 决定当前叠加到哪一步。
输出: 逐步显示 SLM mask、pupil before/after mask、sample intensity、illumination volume、detection PSF、effective PSF；可选保存 npz/png。
用法: python -m marimo edit llspy_light_sheet_to_biobeam_marimo__IN_llspy_slm_pattern__OUT_biobeam_effective_psf_inputs__RUN_marimo_edit.py
依赖: marimo, numpy, matplotlib；当前 CPU fallback 不依赖 NVIDIA；后续可把 propagation 步骤替换成 BioBeam OpenCL BPM。
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="wide")


@app.cell
def _():
    import sys
    from pathlib import Path
    from textwrap import dedent

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return Path, dedent, mo, np, plt, sys


@app.cell
def _(Path, sys):
    SCRIPT_PATH = Path(__file__).resolve()
    PROJECT_ROOT = SCRIPT_PATH.parents[1]
    OUTPUT_ROOT = SCRIPT_PATH.parent / "outputs"
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from llspy_fillchip_shared_math__IN_llspy_gui_params__OUT_slm_pupil_sample_arrays import (
        FillChipParams,
        array_stats,
        compute_biobeam_bridge_outputs,
        log01,
        save_bridge_outputs,
        stats_markdown,
    )

    return (
        FillChipParams,
        OUTPUT_ROOT,
        array_stats,
        compute_biobeam_bridge_outputs,
        log01,
        save_bridge_outputs,
        stats_markdown,
    )


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
            # 02: LLSPY 光片导入 BioBeam 的原理渐进式叠加

            ## LLSM 激发端从光源到 pupil before mask 的物理图像

            这里说的 `point source` 需要先澄清：在 LLSM/LLSPY 的激发端，它不是样品里的荧光点源，也不是 detection PSF 里常说的成像点源。真实硬件里的光源是相干激光。经过扩束、准直和中继光路之后，进入 SLM 的通常可以近似看成一个相干的、空间上比较均匀的入射场。代码里从一个“理想输入场”开始，是 Fourier optics 的简化写法：先假设有一束相干激发光，再让 SLM 只改变它的相位。

            SLM mask 的作用不是直接画出样品里的光片，而是在激发光的相位上写入一个二维图案。LLSPY 的二值 SLM 图通常可以理解成 `0` 和 `pi` 两种相位状态：白/黑像素不是普通亮度，而是让通过这些位置的相干光获得不同相位。因为光是相干的，这些相位差会在后续傅立叶平面里发生干涉，重新分配能量到特定角度/空间频率上。

            `pupil before mask` 对应的是：SLM 后的复振幅场经过一组透镜映射到激发物镜的 pupil plane 之前看到的频谱分布。薄透镜在焦平面上近似做傅立叶变换，所以：

            `SLM complex field = exp(i * SLM phase)`

            `pupil before mask ~= FFT(SLM complex field)`

            这张图展示的是 SLM 相位图把相干激光能量打散到哪些 k-space 位置。换句话说，它不是样品中的真实光强，而是“哪些传播角度/NA 分量将进入激发物镜”的候选频谱。

            后面的 `annular mask` 才对应激发 pupil 中真正允许通过的 NA 环带。它只保留 `Inner NA` 到 `Outer NA` 之间的频率分量，得到 `pupil after mask`。这些被保留下来的环形频率分量再传播/聚焦到样品空间，就形成 Bessel/lattice light sheet 的激发结构。再经过 dither 扫描平均后，才更接近实验里连续的光片。

            这个 notebook 不再一上来展示完整 pipeline，而是按物理链路一层一层叠加：

            1. 先把 LLSPY GUI 参数解析成真实计算量。
            2. 再生成 SLM 上的二值相位图。
            3. 再看 SLM 图经过傅立叶变换后如何进入 pupil plane，并被 annular mask 筛选。
            4. 再看 pupil plane 如何变成 sample plane 光片。
            5. 再把带相位的 pupil field 沿 z 传播成 3D illumination volume。
            6. 再叠加 detection PSF。
            7. 最后才得到 effective PSF。

            你可以把它理解成“每个阶段只多加一个物理模块”，这样不会一开始就被完整代码和完整图像淹没。
            """
        ).strip()
    )
    return


@app.cell(hide_code=True)
def _(np, plt):
    from matplotlib.patches import Circle, Rectangle

    fig_source_to_pupil, _ax = plt.subplots(figsize=(15, 5.2))
    _ax.set_xlim(0, 15)
    _ax.set_ylim(0, 5.2)
    _ax.axis("off")

    def _box(x, y, w, h, label, face="#f7f7f7", edge="#222222"):
        _patch = Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.4)
        _ax.add_patch(_patch)
        _ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
        return _patch

    def _arrow(x0, y0, x1, y1, text=None):
        _ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#222222"},
        )
        if text:
            _ax.text((x0 + x1) / 2, y0 + 0.22, text, ha="center", fontsize=8.5)

    _ax.text(
        0.2,
        4.75,
        "LLSM / LLSPY excitation path: coherent laser phase pattern -> pupil frequency content",
        fontsize=13,
        weight="bold",
        ha="left",
    )

    _box(0.4, 2.0, 1.2, 1.0, "Laser\ncoherent\ninput", face="#ffe9e3")
    _ax.plot([0.7, 1.3], [2.5, 2.5], color="#d33", linewidth=3)
    _ax.plot([0.7, 1.3], [2.35, 2.35], color="#d33", linewidth=1.6)
    _ax.plot([0.7, 1.3], [2.65, 2.65], color="#d33", linewidth=1.6)

    _box(2.1, 1.9, 1.35, 1.2, "Beam expand\ncollimate", face="#edf4ff")
    _box(4.0, 1.55, 1.55, 1.9, "SLM\nphase mask\n0 / pi", face="#f2ecff")
    for _i in range(6):
        for _j in range(8):
            _color = "#222222" if (_i + _j) % 2 else "#ffffff"
            _ax.add_patch(
                Rectangle(
                    (4.18 + 0.14 * _j, 1.78 + 0.22 * _i),
                    0.12,
                    0.18,
                    facecolor=_color,
                    edgecolor="#999999",
                    linewidth=0.25,
                )
            )

    _box(6.15, 1.8, 1.2, 1.4, "Fourier\nlens", face="#eaf8ef")
    _ax.plot([6.75, 6.75], [1.65, 3.35], color="#2d7f46", linewidth=2)
    _ax.plot([6.58, 6.92], [1.65, 3.35], color="#2d7f46", linewidth=1)
    _ax.plot([6.92, 6.58], [1.65, 3.35], color="#2d7f46", linewidth=1)

    _box(8.0, 1.45, 1.8, 2.1, "Pupil before\nannular mask\nFFT(SLM field)", face="#fff7df")
    for _r, _alpha in [(0.18, 0.3), (0.35, 0.45), (0.62, 0.6), (0.85, 0.5)]:
        _ax.add_patch(
            Circle(
                (8.9, 2.5),
                _r,
                fill=False,
                edgecolor="#d98c00",
                linewidth=1.4,
                alpha=_alpha,
            )
        )
    _ax.scatter(
        [8.55, 8.75, 9.02, 9.27, 8.42, 9.34],
        [2.18, 2.92, 2.84, 2.2, 2.62, 2.58],
        s=[28, 18, 34, 20, 16, 22],
        color="#d98c00",
        alpha=0.8,
    )

    _box(10.35, 1.45, 1.65, 2.1, "Annular\nNA mask", face="#eef7f8")
    _ax.add_patch(Circle((11.18, 2.5), 0.76, fill=False, edgecolor="#087f8c", linewidth=5))
    _ax.add_patch(Circle((11.18, 2.5), 0.42, fill=False, edgecolor="#ffffff", linewidth=9))
    _ax.text(11.18, 1.35, "Inner NA < k < Outer NA", ha="center", fontsize=8.5)

    _box(12.55, 1.55, 1.55, 1.9, "Objective\n+ sample", face="#edf4ff")
    _ax.plot([13.0, 13.65], [2.5, 2.5], color="#d33", linewidth=2)
    _ax.plot([13.65, 14.35], [2.12, 2.88], color="#d33", linewidth=1.2)
    _ax.plot([13.65, 14.35], [2.88, 2.12], color="#d33", linewidth=1.2)
    _ax.text(13.9, 3.25, "Bessel / lattice\nlight sheet", ha="center", fontsize=9)

    _arrow(1.6, 2.5, 2.1, 2.5, "uniform phase")
    _arrow(3.45, 2.5, 4.0, 2.5, "write phase")
    _arrow(5.55, 2.5, 6.15, 2.5, "complex field")
    _arrow(7.35, 2.5, 8.0, 2.5, "Fourier transform")
    _arrow(9.8, 2.5, 10.35, 2.5, "NA selection")
    _arrow(12.0, 2.5, 12.55, 2.5, "focus")

    _ax.text(
        0.45,
        0.45,
        "Key idea: the SLM does not draw the light sheet directly. It writes a phase pattern. "
        "The lens converts that phase pattern into k-space content at the pupil. "
        "The annular pupil mask keeps only the NA ring that forms the Bessel/lattice illumination.",
        fontsize=10,
        ha="left",
        va="bottom",
        wrap=True,
    )

    fig_source_to_pupil.tight_layout()
    fig_source_to_pupil
    return


@app.cell
def _(mo):
    w_stage = mo.ui.slider(
        1,
        7,
        step=1,
        value=1,
        label="原理阶段：显示前 N 个物理模块",
        show_value=True,
    )
    w_show_all_controls = mo.ui.checkbox(value=False, label="显示所有参数控制")

    w_preview = mo.ui.slider(
        128, 768, step=64, value=384, label="Simulation pixels", show_value=True
    )
    w_wave_nm = mo.ui.slider(
        405, 640, step=1, value=488, label="Excitation wavelength (nm)", show_value=True
    )
    w_inner_na = mo.ui.slider(
        0.10, 0.90, step=0.005, value=0.500, label="Inner NA", show_value=True
    )
    w_outer_na = mo.ui.slider(
        0.11, 0.95, step=0.005, value=0.550, label="Outer NA", show_value=True
    )
    w_tilt_rad = mo.ui.slider(
        -0.80, 0.80, step=0.01, value=0.00, label="Tilt (rad)", show_value=True
    )
    w_mag = mo.ui.slider(
        80.0, 240.0, step=0.1, value=167.364, label="Mag", show_value=True
    )
    w_crop = mo.ui.slider(
        0.00, 0.60, step=0.005, value=0.220, label="Crop", show_value=True
    )
    w_fudge = mo.ui.slider(
        0.80,
        1.20,
        step=0.005,
        value=0.970,
        label="Auto spacing factor",
        show_value=True,
    )
    w_fillchip = mo.ui.slider(
        0.10, 1.00, step=0.01, value=0.95, label="Fill chip fraction", show_value=True
    )
    w_z_half = mo.ui.slider(
        3.0,
        60.0,
        step=1.0,
        value=12.0,
        label="Propagation z half range for PSF window (um)",
        show_value=True,
    )
    w_nz = mo.ui.slider(
        21, 161, step=10, value=81, label="Number of z planes", show_value=True
    )
    w_medium_ri = mo.ui.slider(
        1.00, 1.52, step=0.01, value=1.33, label="Medium RI", show_value=True
    )
    w_em_nm = mo.ui.slider(
        500, 700, step=1, value=525, label="Emission wavelength (nm)", show_value=True
    )
    w_det_na = mo.ui.slider(
        0.30,
        1.20,
        step=0.01,
        value=0.90,
        label="Detection NA default",
        show_value=True,
    )
    w_dither = mo.ui.checkbox(value=True, label="Dither excitation into light sheet")
    w_biobeam_detection = mo.ui.checkbox(value=True, label="Use BioBeam detection PSF")
    w_save = mo.ui.checkbox(value=False, label="Save outputs")
    return (
        w_crop,
        w_det_na,
        w_dither,
        w_biobeam_detection,
        w_em_nm,
        w_fillchip,
        w_fudge,
        w_inner_na,
        w_mag,
        w_medium_ri,
        w_nz,
        w_outer_na,
        w_preview,
        w_save,
        w_show_all_controls,
        w_stage,
        w_tilt_rad,
        w_wave_nm,
        w_z_half,
    )


@app.cell
def _():
    PIPELINE_STEPS = [
        {
            "stage": 1,
            "title": "1. 参数解析",
            "plain": "先不要看图。先把 GUI 参数变成计算真正使用的量，例如 spacing、# Beams、pupil ring 的 k-space 半径。",
            "formula": "spacing = factor * wavelength / Inner NA",
        },
        {
            "stage": 2,
            "title": "2. SLM 二值相位图",
            "plain": "LLSPY 先在 SLM 上生成黑白二值图。黑和白对应 0/pi 相位，不是普通亮度图。",
            "formula": "continuous SLM field -> crop threshold -> binary 0/pi phase",
        },
        {
            "stage": 3,
            "title": "3. Pupil plane + annular mask",
            "plain": "SLM 相位图做傅立叶变换后到达 pupil plane。annular mask 只保留 Inner NA 到 Outer NA 之间的环形频率。",
            "formula": "pupil_after_mask = FFT(SLM phase) * annular_mask",
        },
        {
            "stage": 4,
            "title": "4. Sample plane 光片",
            "plain": "筛选后的 pupil field 再做一次傅立叶变换，就得到样品平面的 LLSPY 光片强度。",
            "formula": "sample_intensity = |FFT(pupil_after_mask)|^2",
        },
        {
            "stage": 5,
            "title": "5. 3D illumination volume",
            "plain": "前面只是一张 sample plane 图。为了做 PSF，需要先沿激发物镜光轴传播，再用 dither 扫描积分成光片。",
            "formula": "illumination = dither(|FFT(pupil * exp(i kz z_exc))|^2)",
        },
        {
            "stage": 6,
            "title": "6. Detection PSF",
            "plain": "显微镜最后看到的不是 illumination 本身，还要乘上检测通道的 PSF。优先使用 BioBeam Debye PSF；不可用时退回高斯近似。",
            "formula": "detection_psf = BioBeam focus_field_beam(...) or Gaussian fallback",
        },
        {
            "stage": 7,
            "title": "7. Effective PSF",
            "plain": "最终 effective PSF 是先把 excitation light sheet 转到 detection objective 坐标系后，再和 detection PSF 相乘。",
            "formula": "effective_psf = excitation_in_detection_coords * detection_psf",
        },
    ]
    return (PIPELINE_STEPS,)


@app.cell
def _(
    FillChipParams,
    w_crop,
    w_fillchip,
    w_fudge,
    w_inner_na,
    w_mag,
    w_outer_na,
    w_tilt_rad,
    w_wave_nm,
):
    outer_na = max(w_outer_na.value, w_inner_na.value + 0.005)
    params = FillChipParams(
        wavelength_um=w_wave_nm.value / 1000.0,
        inner_na=w_inner_na.value,
        outer_na=outer_na,
        tilt_rad=w_tilt_rad.value,
        mag=w_mag.value,
        crop=w_crop.value,
        auto_spacing=True,
        fudge=w_fudge.value,
        auto_fill=True,
        fillchip=w_fillchip.value,
    )
    return (params,)


@app.cell
def _(
    compute_biobeam_bridge_outputs,
    mo,
    params,
    w_biobeam_detection,
    w_det_na,
    w_dither,
    w_em_nm,
    w_medium_ri,
    w_nz,
    w_preview,
    w_z_half,
):
    with mo.status.spinner("Computing LLSPY -> BioBeam bridge data..."):
        bridge = compute_biobeam_bridge_outputs(
            params,
            preview_pixels=int(w_preview.value),
            z_half_um=w_z_half.value,
            nz=int(w_nz.value),
            medium_ri=w_medium_ri.value,
            emission_wavelength_um=w_em_nm.value / 1000.0,
            detection_na=w_det_na.value,
            dither_excitation=w_dither.value,
            prefer_biobeam_detection=w_biobeam_detection.value,
        )
    return (bridge,)


@app.cell
def _(PIPELINE_STEPS, bridge, w_stage):
    stage_n = int(w_stage.value)
    ll = bridge["llspy_like"]
    resolved = bridge["resolved_params"]
    current_step = PIPELINE_STEPS[stage_n - 1]
    active_steps = PIPELINE_STEPS[:stage_n]
    hidden_steps = PIPELINE_STEPS[stage_n:]
    return active_steps, current_step, hidden_steps, ll, resolved, stage_n


@app.cell
def _(
    mo,
    stage_n,
    w_biobeam_detection,
    w_crop,
    w_det_na,
    w_dither,
    w_em_nm,
    w_fillchip,
    w_fudge,
    w_inner_na,
    w_mag,
    w_medium_ri,
    w_nz,
    w_outer_na,
    w_preview,
    w_save,
    w_show_all_controls,
    w_stage,
    w_tilt_rad,
    w_wave_nm,
    w_z_half,
):
    all_controls = [
        w_preview,
        w_wave_nm,
        w_inner_na,
        w_outer_na,
        w_fudge,
        w_fillchip,
        w_crop,
        w_mag,
        w_tilt_rad,
        w_z_half,
        w_nz,
        w_medium_ri,
        w_em_nm,
        w_det_na,
        w_dither,
        w_biobeam_detection,
        w_save,
    ]
    stage_controls = {
        1: [w_preview, w_wave_nm, w_inner_na, w_outer_na, w_fudge, w_fillchip],
        2: [w_crop, w_mag, w_tilt_rad],
        3: [w_inner_na, w_outer_na],
        4: [w_wave_nm, w_fudge, w_fillchip],
        5: [w_z_half, w_nz, w_medium_ri, w_dither],
        6: [w_em_nm, w_det_na, w_biobeam_detection],
        7: [w_save],
    }
    visible_controls = all_controls if w_show_all_controls.value else stage_controls[stage_n]
    control_columns = [
        mo.vstack(chunk)
        for chunk in (
            visible_controls[:5],
            visible_controls[5:10],
            visible_controls[10:],
        )
        if chunk
    ]
    controls = mo.vstack(
        [
            mo.md("## 原理阶段"),
            w_stage,
            w_show_all_controls,
            mo.md("## 当前阶段相关控制"),
            mo.hstack(control_columns),
        ]
    )
    return (controls,)


@app.cell
def _(active_steps, current_step, dedent, hidden_steps, mo, resolved, stage_n):
    active_table = [
        {
            "阶段": step["stage"],
            "模块": step["title"],
            "核心公式/关系": step["formula"],
        }
        for step in active_steps
    ]
    hidden_names = ", ".join(step["title"] for step in hidden_steps) or "无"
    stage_explanation = mo.md(
        dedent(
            f"""
            ## 当前叠加到第 {stage_n}/7 阶段：{current_step["title"]}

            新加入的物理模块：

            **{current_step["plain"]}**

            这一阶段的核心关系：

            `{current_step["formula"]}`

            还没叠加的模块：{hidden_names}

            当前关键派生量：

            - `spacing = {resolved["spacing_um"]:.4f} um`
            - `# Beams = {resolved["n_beams"]}`
            - `phase terms = {resolved["total_phase_terms"]}`
            - `projected SLM pixel = {resolved["projected_pixel_um"]:.4f} um/pixel`
            - `annular pupil k range = {resolved["annulus_inner_rad_per_um"]:.3f} ~ {resolved["annulus_outer_rad_per_um"]:.3f} rad/um`
            """
        ).strip()
    )
    return active_table, stage_explanation


@app.cell
def _(array_stats, bridge, ll, stage_n, stats_markdown):
    _resolved = bridge["resolved_params"]
    _resolved_key_numbers = [
        _resolved["wavelength_um"],
        _resolved["inner_na"],
        _resolved["outer_na"],
        _resolved["spacing_um"],
        _resolved["n_beams"],
        _resolved["total_phase_terms"],
        _resolved["projected_pixel_um"],
        _resolved["annulus_inner_rad_per_um"],
        _resolved["annulus_outer_rad_per_um"],
    ]
    rows = [
        array_stats(
            "resolved LLSPY parameters",
            _resolved_key_numbers,
            "GUI 参数被解析成 spacing、# Beams、pupil ring k-space 半径等真实计算量。",
        )
    ]
    if stage_n >= 2:
        rows.append(
            array_stats(
                "binary SLM mask",
                ll["slm_binary_mask"],
                "SLM 上实际写入的 0/pi 二值相位图。",
            )
        )
    if stage_n >= 3:
        rows.extend(
            [
                array_stats(
                    "pupil impinging intensity",
                    ll["pupil_plane_impinging_intensity"],
                    "SLM 相位图傅立叶变换后，尚未被 annular mask 筛选前的 pupil plane 强度。",
                ),
                array_stats(
                    "pupil after annular mask",
                    ll["pupil_plane_intensity"],
                    "只保留 Inner NA 到 Outer NA 环形区域之后的 pupil plane 强度。",
                ),
                array_stats(
                    "complex pupil field",
                    ll["complex_pupil_field_after_mask"],
                    "带相位的 pupil field；这是后续 3D propagation 的入口。",
                ),
            ]
        )
    if stage_n >= 4:
        rows.append(
            array_stats(
                "sample intensity",
                ll["sample_plane_intensity"],
                "LLSPY Fourier 链路在样品平面得到的光片强度。",
            )
        )
    if stage_n >= 5:
        rows.append(
            array_stats(
                "illumination volume",
                bridge["illumination_volume_cpu"],
                "complex pupil field 沿 z 传播得到的 3D excitation light sheet。",
            )
        )
    if stage_n >= 6:
        rows.append(
                array_stats(
                    "detection PSF",
                    bridge["detection_psf_cpu"],
                    f"detection PSF model: {bridge['detection_model']}。",
                )
            )
    if stage_n >= 7:
        rows.append(
            array_stats(
                "effective PSF",
                bridge["effective_psf_cpu"],
                "illumination volume 和 detection PSF 相乘后的最终 effective PSF。",
            )
        )
    stats_md = stats_markdown(rows)
    return rows, stats_md


@app.cell
def _(bridge, ll, log01, mo, np, plt, resolved, stage_n):
    _illum = bridge["illumination_volume_cpu"]
    _det = bridge["detection_psf_cpu"]
    _eff = bridge["effective_psf_cpu"]
    _yc = _illum.shape[1] // 2
    _x_um = bridge["x_um"]
    _z_um = bridge["z_um"]
    _xz_extent = [float(_x_um[0]), float(_x_um[-1]), float(_z_um[0]), float(_z_um[-1])]

    panels = [
        {
            "stage": 1,
            "title": "1. Resolved parameters",
            "kind": "text",
            "text": (
                f"wavelength = {resolved['wavelength_um']:.3f} um\n"
                f"Inner/Outer NA = {resolved['inner_na']:.3f} / {resolved['outer_na']:.3f}\n"
                f"spacing = {resolved['spacing_um']:.4f} um\n"
                f"# Beams = {resolved['n_beams']}\n"
                f"phase terms = {resolved['total_phase_terms']}\n"
                f"projected pixel = {resolved['projected_pixel_um']:.4f} um"
            ),
        }
    ]
    if stage_n >= 2:
        panels.append(
            {
                "stage": 2,
                "title": "2. Binary SLM mask",
                "kind": "image",
                "data": ll["slm_binary_mask"],
                "cmap": "gray",
                "aspect": "auto",
                "xlabel": "SLM x pixel",
                "ylabel": "SLM y pixel",
            }
        )
    if stage_n >= 3:
        panels.extend(
            [
                {
                    "stage": 3,
                    "title": "3a log. Pupil before mask",
                    "kind": "image",
                    "data": log01(ll["pupil_plane_impinging_intensity"]),
                    "cmap": "inferno",
                    "aspect": "equal",
                    "clim": "auto",
                    "xlabel": "kx pixel",
                    "ylabel": "ky pixel",
                },
                {
                    "stage": 3,
                    "title": "3b log. Pupil after annular mask",
                    "kind": "image",
                    "data": log01(ll["pupil_plane_intensity"]),
                    "cmap": "inferno",
                    "aspect": "equal",
                    "clim": "auto",
                    "xlabel": "kx pixel",
                    "ylabel": "ky pixel",
                },
                {
                    "stage": 3,
                    "title": "3a. Pupil before mask",
                    "kind": "image",
                    "data": ll["pupil_plane_impinging_intensity"],
                    "cmap": "inferno",
                    "aspect": "equal",
                    "xlabel": "kx pixel",
                    "ylabel": "ky pixel",
                },
                {
                    "stage": 3,
                    "title": "3b. Pupil after annular mask",
                    "kind": "image",
                    "data": ll["pupil_plane_intensity"],
                    "cmap": "inferno",
                    "aspect": "equal",
                    "xlabel": "kx pixel",
                    "ylabel": "ky pixel",
                },
            ]
        )
    if stage_n >= 4:
        panels.append(
            {
                "stage": 4,
                "title": "4. LLSPY sample intensity",
                "kind": "image",
                "data": ll["sample_plane_intensity"],
                "cmap": "viridis",
                "aspect": "auto",
                "xlabel": "sample x pixel",
                "ylabel": "sample y pixel",
            }
        )
    if stage_n >= 5:
        panels.append(
            {
                "stage": 5,
                "title": "5. Illumination volume xz",
                "kind": "image",
                "data": _illum[:, _yc, :],
                "cmap": "magma",
                "aspect": "auto",
                "extent": _xz_extent,
                "xlabel": "x (um)",
                "ylabel": "z (um)",
            }
        )
    if stage_n >= 6:
        panels.append(
            {
                "stage": 6,
                "title": "6. Detection PSF xz",
                "kind": "image",
                "data": _det[:, _yc, :],
                "cmap": "magma",
                "aspect": "auto",
                "extent": _xz_extent,
                "xlabel": "x (um)",
                "ylabel": "z (um)",
            }
        )
    if stage_n >= 7:
        panels.append(
            {
                "stage": 7,
                "title": "7. Effective PSF xz",
                "kind": "image",
                "data": _eff[:, _yc, :],
                "cmap": "magma",
                "aspect": "auto",
                "extent": _xz_extent,
                "xlabel": "x (um)",
                "ylabel": "z (um)",
            }
        )

    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig_pipeline, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    fig_pipeline.suptitle(f"Progressive LLSPY -> BioBeam pipeline, stage {stage_n}/7", fontsize=12)

    for _ax, panel in zip(axes, panels):
        if panel["kind"] == "text":
            _ax.axis("off")
            _ax.text(0.02, 0.96, panel["text"], va="top", family="monospace", fontsize=10)
            _ax.set_title(panel["title"])
        else:
            kwargs = {
                "cmap": panel["cmap"],
                "origin": "lower",
                "aspect": panel["aspect"],
            }
            if panel.get("clim") != "auto":
                kwargs["vmin"] = 0.0
                kwargs["vmax"] = 1.0
            if "extent" in panel:
                kwargs["extent"] = panel["extent"]
            _ax.imshow(panel["data"], **kwargs)
            _ax.set_title(panel["title"])
            _ax.set_xlabel(panel["xlabel"])
            _ax.set_ylabel(panel["ylabel"])
        _ax.tick_params(labelsize=8)

    for _ax in axes[n_panels:]:
        _ax.axis("off")
    fig_pipeline.tight_layout()

    zoom_options = [panel["title"] for panel in panels]
    w_zoom_panel = mo.ui.dropdown(
        options=zoom_options,
        value=zoom_options[-1],
        label="放大查看",
    )
    return fig_pipeline, panels, w_zoom_panel


@app.cell
def _(np, panels, plt, w_zoom_panel):
    selected_panel = next(
        (panel for panel in panels if panel["title"] == w_zoom_panel.value),
        panels[-1],
    )
    fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(10, 8))
    if selected_panel["kind"] == "text":
        ax_zoom.axis("off")
        ax_zoom.text(
            0.02,
            0.98,
            selected_panel["text"],
            va="top",
            family="monospace",
            fontsize=13,
        )
    else:
        _zoom_kwargs = {
            "cmap": selected_panel["cmap"],
            "origin": "lower",
            "aspect": selected_panel["aspect"],
        }
        if selected_panel.get("clim") != "auto":
            _zoom_kwargs["vmin"] = 0.0
            _zoom_kwargs["vmax"] = 1.0
        if "extent" in selected_panel:
            _zoom_kwargs["extent"] = selected_panel["extent"]
        image = ax_zoom.imshow(np.asarray(selected_panel["data"]), **_zoom_kwargs)
        fig_zoom.colorbar(image, ax=ax_zoom, fraction=0.046, pad=0.04)
        ax_zoom.set_xlabel(selected_panel["xlabel"])
        ax_zoom.set_ylabel(selected_panel["ylabel"])
    ax_zoom.set_title(selected_panel["title"])
    fig_zoom.tight_layout()
    return (fig_zoom,)


@app.cell
def _(bridge, np, plt, stage_n):
    _illum = bridge["illumination_volume_cpu"]
    _det = bridge["detection_psf_cpu"]
    _eff = bridge["effective_psf_cpu"]
    _zc = _illum.shape[0] // 2
    _yc = _illum.shape[1] // 2
    _xc = _illum.shape[2] // 2
    _x_um = bridge["x_um"]
    _z_um = bridge["z_um"]

    fig_profiles, axes_profiles = plt.subplots(1, 2, figsize=(12, 4))
    if stage_n < 5:
        for _ax in axes_profiles:
            _ax.axis("off")
        axes_profiles[0].text(
            0.02,
            0.9,
            "Profiles start at stage 5.\nBefore stage 5, we have not built a 3D illumination volume yet.",
            va="top",
            fontsize=11,
        )
    else:
        axes_profiles[0].plot(_x_um, _illum[_zc, _yc, :], label="illumination")
        if stage_n >= 6:
            axes_profiles[0].plot(_x_um, _det[_zc, _yc, :], label="detection")
        if stage_n >= 7:
            axes_profiles[0].plot(_x_um, _eff[_zc, _yc, :], label="effective")
        axes_profiles[0].set_title("Center z/y profile along x")
        axes_profiles[0].set_xlabel("x (um)")
        axes_profiles[0].set_ylabel("normalized intensity")
        axes_profiles[0].set_xlim(float(np.min(_x_um)), float(np.max(_x_um)))
        axes_profiles[0].legend(fontsize=8)

        axes_profiles[1].plot(_z_um, _illum[:, _yc, _xc], label="illumination")
        if stage_n >= 6:
            axes_profiles[1].plot(_z_um, _det[:, _yc, _xc], label="detection")
        if stage_n >= 7:
            axes_profiles[1].plot(_z_um, _eff[:, _yc, _xc], label="effective")
        axes_profiles[1].set_title("Center x/y profile along z")
        axes_profiles[1].set_xlabel("z (um)")
        axes_profiles[1].set_ylabel("normalized intensity")
        axes_profiles[1].legend(fontsize=8)
    fig_profiles.tight_layout()
    return (fig_profiles,)


@app.cell
def _(OUTPUT_ROOT, bridge, fig_pipeline, fig_profiles, save_bridge_outputs, stage_n, w_save):
    saved_message = "未保存。第 7 阶段勾选 `Save outputs` 后会写入 outputs 文件夹。"
    if w_save.value and stage_n < 7:
        saved_message = "还没到第 7 阶段，所以暂不保存。先把 effective PSF 叠加出来。"
    if w_save.value and stage_n >= 7:
        outdir = save_bridge_outputs(OUTPUT_ROOT, bridge)
        fig_pipeline.savefig(outdir / "overview__progressive_llspy_to_biobeam.png", dpi=160, bbox_inches="tight")
        fig_profiles.savefig(outdir / "profiles__progressive_llspy_to_biobeam.png", dpi=160, bbox_inches="tight")
        saved_message = f"已保存到 `{outdir}`"
    return (saved_message,)


@app.cell(hide_code=True)
def _(
    active_table,
    bridge,
    controls,
    fig_pipeline,
    fig_profiles,
    fig_zoom,
    mo,
    saved_message,
    stage_explanation,
    stats_md,
    w_zoom_panel,
):
    mo.vstack(
        [
            controls,
            stage_explanation,
            mo.ui.table(active_table, label="已经叠加的物理模块"),
            fig_pipeline,
            w_zoom_panel,
            fig_zoom,
            fig_profiles,
            mo.md(f"## 保存状态\n\n{saved_message}"),
            mo.md("## Data Shape / Range 检查表"),
            mo.md(stats_md),
            mo.md(
                "## 关键提醒\n\n"
                f"当前 detection PSF model: `{bridge['detection_model']}`。"
                "effective PSF 已先把 dither 后的 excitation light sheet 转到 detection objective 坐标系再相乘。"
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
