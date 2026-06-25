"""
用途: 用 marimo 做渐进式 LLSPY Square Lattice Fill Chip 参数教学，从 1 个参数逐步增加到 10 个参数。
输入: LLSPY GUI 默认参数；教学阶段 slider 决定当前开放前几个参数。
输出: 每一阶段对应的 SLM mask、pupil plane、sample plane、参数解释和 data shape/range 检查表。
用法: python -m marimo edit llspy_parameters_tutorial_marimo__IN_llspy_gui_defaults__OUT_interactive_parameter_figures__RUN_marimo_edit.py
依赖: marimo, numpy, matplotlib；共享模块 llspy_fillchip_shared_math__IN_llspy_gui_params__OUT_slm_pupil_sample_arrays.py。
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
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from llspy_fillchip_shared_math__IN_llspy_gui_params__OUT_slm_pupil_sample_arrays import (
        FillChipParams,
        array_stats,
        compute_fill_chip_outputs,
        log01,
        resolve_fill_chip_params,
        stats_markdown,
    )

    return (
        FillChipParams,
        array_stats,
        compute_fill_chip_outputs,
        log01,
        resolve_fill_chip_params,
        stats_markdown,
    )


@app.cell(hide_code=True)
def _(dedent, mo):
    mo.md(
        dedent(
            r"""
        # LLSPY `Square Lattice, Fill Chip` 渐进式参数教学

        这个 notebook 不再一次性把所有参数都摆出来，而是像课程一样逐步增加：

        - 阶段 1：只出现 1 个参数。
        - 阶段 2：出现前 2 个参数。
        - 阶段 3：出现前 3 个参数。
        - 以此类推，直到阶段 10 才看到完整 10 个参数。

        没出现的参数会固定在 LLSPY Fill Chip 默认值上，所以你可以清楚地看到“新增的这个参数”到底改变了什么。
        """
        ).strip()
    )
    return


@app.cell
def _(mo):
    w_stage = mo.ui.slider(
        1,
        10,
        step=1,
        value=1,
        label="教学阶段：显示前 N 个参数",
        show_value=True,
    )

    w_wave_nm = mo.ui.slider(
        405, 640, step=1, value=488, label="1. Wavelength (nm)", show_value=True
    )
    w_inner_na = mo.ui.slider(
        0.10, 0.90, step=0.005, value=0.500, label="2. Inner NA", show_value=True
    )
    w_outer_na = mo.ui.slider(
        0.11, 0.95, step=0.005, value=0.550, label="3. Outer NA", show_value=True
    )
    w_fudge = mo.ui.slider(
        0.80,
        1.20,
        step=0.005,
        value=0.970,
        label="4. Auto spacing factor",
        show_value=True,
    )
    w_n_beams = mo.ui.slider(
        1, 120, step=1, value=53, label="5. # Beams", show_value=True
    )
    w_crop = mo.ui.slider(
        0.00, 0.60, step=0.005, value=0.220, label="6. Crop", show_value=True
    )
    w_tilt_rad = mo.ui.slider(
        -0.80, 0.80, step=0.01, value=0.00, label="7. Tilt (rad)", show_value=True
    )
    w_mag = mo.ui.slider(
        80.0, 240.0, step=0.1, value=167.364, label="8. Mag", show_value=True
    )
    w_shift_x = mo.ui.slider(
        -5.0, 5.0, step=0.1, value=0.0, label="9. Shift X (um)", show_value=True
    )
    w_shift_y = mo.ui.slider(
        -5.0, 5.0, step=0.1, value=0.0, label="10. Shift Y (um)", show_value=True
    )
    return (
        w_crop,
        w_fudge,
        w_inner_na,
        w_mag,
        w_n_beams,
        w_outer_na,
        w_shift_x,
        w_shift_y,
        w_stage,
        w_tilt_rad,
        w_wave_nm,
    )


@app.cell
def _(
    w_crop,
    w_fudge,
    w_inner_na,
    w_mag,
    w_n_beams,
    w_outer_na,
    w_shift_x,
    w_shift_y,
    w_tilt_rad,
    w_wave_nm,
):
    DEFAULTS = {
        "wavelength_nm": 488,
        "inner_na": 0.500,
        "outer_na": 0.550,
        "fudge": 0.970,
        "n_beams": 53,
        "crop": 0.220,
        "tilt_rad": 0.0,
        "mag": 167.364,
        "shift_x_um": 0.0,
        "shift_y_um": 0.0,
    }

    PARAMETER_STEPS = [
        {
            "key": "wavelength_nm",
            "title": "1. Wavelength",
            "widget": w_wave_nm,
            "plain": "激发光的颜色。488 nm 就是你数据里 488 激光。",
            "watch": "主要观察 sample 图案整体尺度，以及 spacing 公式里的 wavelength。",
        },
        {
            "key": "inner_na",
            "title": "2. Inner NA",
            "widget": w_inner_na,
            "plain": "环形 pupil 的内半径。内圈越大，低角度光越少。",
            "watch": "主要观察 pupil plane 中黑洞内圈大小，以及 auto spacing 的变化。",
        },
        {
            "key": "outer_na",
            "title": "3. Outer NA",
            "widget": w_outer_na,
            "plain": "环形 pupil 的外半径。外圈越大，允许更高角度的光通过。",
            "watch": "主要观察 pupil ring 的厚度，sample 光片会更细但旁瓣也可能更明显。",
        },
        {
            "key": "fudge",
            "title": "4. Auto spacing factor",
            "widget": w_fudge,
            "plain": "LLSPY 里用来微调间距的系数。公式是 spacing = factor * wavelength / Inner NA。",
            "watch": "主要观察 sample 平面横向条纹间距。",
        },
        {
            "key": "n_beams",
            "title": "5. # Beams",
            "widget": w_n_beams,
            "plain": "参与相干叠加的 Bessel beam 数量。代码里实际相干项数是 2*n - 1。",
            "watch": "主要观察 SLM 条纹数量和 sample 中 lattice 覆盖范围。",
        },
        {
            "key": "crop",
            "title": "6. Crop",
            "widget": w_crop,
            "plain": "把连续 SLM 场变成黑白二值相位图之前的阈值。",
            "watch": "主要观察 binary SLM mask 的黑白条纹宽度和断裂程度。",
        },
        {
            "key": "tilt_rad",
            "title": "7. Tilt",
            "widget": w_tilt_rad,
            "plain": "旋转 lattice 方向。0 rad 就是不旋转。",
            "watch": "主要观察 SLM 条纹方向和 sample lattice 方向。",
        },
        {
            "key": "mag",
            "title": "8. Mag",
            "widget": w_mag,
            "plain": "SLM 到样品面的缩放倍率。倍率越大，SLM 单像素投影到样品侧越小。",
            "watch": "主要观察 projected pixel size 和 Fill Chip 覆盖尺度。",
        },
        {
            "key": "shift_x_um",
            "title": "9. Shift X",
            "widget": w_shift_x,
            "plain": "给 pupil 加 x 方向相位斜坡，让样品平面图案平移。",
            "watch": "主要观察 sample 图案在 x 方向的位置变化。",
        },
        {
            "key": "shift_y_um",
            "title": "10. Shift Y",
            "widget": w_shift_y,
            "plain": "给 pupil 加 y 方向相位斜坡，让样品平面图案平移。",
            "watch": "主要观察 sample 图案在 y 方向的位置变化。",
        },
    ]
    return DEFAULTS, PARAMETER_STEPS


@app.cell
def _(DEFAULTS, FillChipParams, PARAMETER_STEPS, resolve_fill_chip_params, w_stage):
    stage_n = int(w_stage.value)

    def value_for(key):
        for index, step in enumerate(PARAMETER_STEPS, start=1):
            if step["key"] == key:
                return step["widget"].value if stage_n >= index else DEFAULTS[key]
        raise KeyError(key)

    outer_na = max(float(value_for("outer_na")), float(value_for("inner_na")) + 0.005)
    params = FillChipParams(
        wavelength_um=float(value_for("wavelength_nm")) / 1000.0,
        inner_na=float(value_for("inner_na")),
        outer_na=outer_na,
        shift_x_um=float(value_for("shift_x_um")),
        shift_y_um=float(value_for("shift_y_um")),
        tilt_rad=float(value_for("tilt_rad")),
        mag=float(value_for("mag")),
        crop=float(value_for("crop")),
        auto_spacing=True,
        fudge=float(value_for("fudge")),
        auto_fill=False,
        n_beams=int(value_for("n_beams")),
        fillchip=0.950,
    )
    resolved = resolve_fill_chip_params(params)
    current_step = PARAMETER_STEPS[stage_n - 1]
    active_steps = PARAMETER_STEPS[:stage_n]
    hidden_steps = PARAMETER_STEPS[stage_n:]
    return active_steps, current_step, hidden_steps, params, resolved, stage_n


@app.cell
def _(active_steps, current_step, dedent, mo, stage_n, w_stage):
    active_controls = [
        mo.vstack(
            [
                mo.md(f"**{step['title']}**"),
                step["widget"],
                mo.md(f"小白解释：{step['plain']}"),
            ]
        )
        for step in active_steps
    ]

    lesson_panel = mo.vstack(
        [
            mo.md("## 教学阶段"),
            w_stage,
            mo.md(
                dedent(
                    f"""
                当前是 **阶段 {stage_n}/10**。

                这一阶段新加入的参数是：**{current_step["title"]}**。

                观察重点：{current_step["watch"]}
                """
                ).strip()
            ),
            mo.md("## 当前开放的参数"),
            mo.vstack(active_controls),
        ]
    )
    return (lesson_panel,)


@app.cell
def _(compute_fill_chip_outputs, mo, params):
    with mo.status.spinner("Computing current teaching stage..."):
        outputs = compute_fill_chip_outputs(
            params,
            engine="fast_numpy_llspy_like",
            preview_pixels=384,
        )
    return (outputs,)


@app.cell
def _(array_stats, outputs, stats_markdown):
    rows = [
        array_stats(
            "binary SLM mask",
            outputs["slm_binary_mask"],
            "写到 SLM 芯片上的 0/pi 二值相位图；这是硬件最终看到的黑白图案。",
        ),
        array_stats(
            "pupil intensity after mask",
            outputs["pupil_plane_intensity"],
            "SLM 相位图经过傅立叶变换，再通过 Inner NA 和 Outer NA 环形 mask 后的 pupil plane 强度。",
        ),
        array_stats(
            "sample intensity",
            outputs["sample_plane_intensity"],
            "pupil plane 再傅立叶变换后，在样品平面形成的 lattice light sheet 强度。",
        ),
    ]
    stats_md = stats_markdown(rows)
    return (stats_md,)


@app.cell
def _(log01, outputs, plt, resolved, stage_n):
    slm = outputs["slm_binary_mask"]
    pupil = outputs["pupil_plane_intensity"]
    sample = outputs["sample_plane_intensity"]
    cy = sample.shape[0] // 2
    py = pupil.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"Stage {stage_n}/10: progressive LLSPY Fill Chip teaching",
        fontsize=12,
    )

    axes[0, 0].imshow(slm, cmap="gray", origin="lower", aspect="auto")
    axes[0, 0].set_title("1. Binary SLM mask")
    axes[0, 0].set_xlabel("SLM x pixel")
    axes[0, 0].set_ylabel("SLM y pixel")

    axes[0, 1].imshow(log01(pupil), cmap="inferno", origin="lower")
    axes[0, 1].set_title("2. Intensity After Mask (log)")
    axes[0, 1].set_xlabel("pupil kx pixel")
    axes[0, 1].set_ylabel("pupil ky pixel")

    axes[0, 2].imshow(log01(sample), cmap="viridis", origin="lower", aspect="auto")
    axes[0, 2].set_title("3. Intensity At Sample (log)")
    axes[0, 2].set_xlabel("sample x pixel")
    axes[0, 2].set_ylabel("sample y pixel")

    axes[1, 0].plot(sample[cy], lw=1.2)
    axes[1, 0].set_title("Sample center-row profile")
    axes[1, 0].set_xlabel("x pixel")
    axes[1, 0].set_ylabel("normalized intensity")

    axes[1, 1].plot(pupil[py], lw=1.2)
    axes[1, 1].set_title("Pupil center-row profile")
    axes[1, 1].set_xlabel("kx pixel")
    axes[1, 1].set_ylabel("normalized intensity")

    formula_text = (
        "当前派生量\n\n"
        f"spacing = factor * wavelength / inner_NA\n"
        f"        = {resolved['fudge']:.3f} * {resolved['wavelength_um']:.3f} / {resolved['inner_na']:.3f}\n"
        f"        = {resolved['spacing_um']:.4f} um\n\n"
        f"# Beams = {resolved['n_beams']}\n"
        f"phase terms = 2*n - 1 = {resolved['total_phase_terms']}\n\n"
        f"projected SLM pixel = pixel / mag\n"
        f"                    = {resolved['projected_pixel_um']:.4f} um/pixel\n\n"
        f"annulus k range = {resolved['annulus_inner_rad_per_um']:.3f}"
        f" ~ {resolved['annulus_outer_rad_per_um']:.3f} rad/um"
    )
    axes[1, 2].axis("off")
    axes[1, 2].text(0.02, 0.95, formula_text, va="top", family="monospace")

    fig.tight_layout()
    return (fig,)


@app.cell
def _(active_steps, dedent, hidden_steps, mo, resolved, stage_n):
    active_table = [
        {
            "阶段": index,
            "参数": step["title"],
            "当前值": step["widget"].value,
            "小白解释": step["plain"],
        }
        for index, step in enumerate(active_steps, start=1)
    ]
    hidden_names = ", ".join(step["title"] for step in hidden_steps) or "无"
    teaching_text = mo.md(
        dedent(
            f"""
        ## 当前课程状态

        - 现在只开放了前 `{stage_n}` 个参数。
        - 还没开放的参数：{hidden_names}
        - 这些没开放的参数仍然使用默认值，不会干扰你理解当前新增参数。

        这里先把 `# Beams` 做成手动 slider，目的是让你单独观察它的影响。
        LLSPY GUI 的 Fill Chip preset 也可以自动算出默认 `# Beams = 53`。

        当前关键派生结果：

        - `spacing = {resolved["spacing_um"]:.4f} um`
        - `# Beams = {resolved["n_beams"]}`
        - `phase terms = {resolved["total_phase_terms"]}`
        - `projected SLM pixel = {resolved["projected_pixel_um"]:.4f} um/pixel`
        """
        ).strip()
    )
    return active_table, teaching_text


@app.cell(hide_code=True)
def _(active_table, dedent, fig, lesson_panel, mo, stats_md, teaching_text):
    mo.vstack(
        [
            lesson_panel,
            teaching_text,
            mo.ui.table(active_table, label="已经开放的参数"),
            fig,
            mo.md("## Data Shape / Range 检查表"),
            mo.md(stats_md),
            mo.md(
                dedent(
                    r"""
                ## 怎么学习这一页

                1. 先停在阶段 1，只动第一个参数，看三张图怎么变。
                2. 再切到阶段 2，只新动第二个参数，比较它和第一个参数的区别。
                3. 一直重复到阶段 10。
                4. 如果某一步看不懂，先不要继续往后加参数。
                """
                ).strip()
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
