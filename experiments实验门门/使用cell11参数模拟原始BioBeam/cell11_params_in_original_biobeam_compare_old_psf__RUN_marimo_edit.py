"""
04: Run the original BioBeam Bessel lattice light-sheet PSF workflow with
cell11 settings, then compare the generated effective PSF with the old TIFF.

Run:
    python -m marimo edit cell11_params_in_original_biobeam_compare_old_psf__RUN_marimo_edit.py
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="wide")


@app.cell
def _():
    import configparser
    import re
    import sys
    from pathlib import Path
    from textwrap import dedent

    if sys.version_info >= (3, 12) and not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser

    _here = Path(__file__).resolve()
    _repo_root = next(
        (_parent for _parent in [_here, *_here.parents] if (_parent / "biobeam" / "core" / "focus_field_beam.py").exists()),
        Path(r"d:\codes\biobeam"),
    )
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import tifffile
    from biobeam.core.focus_field_beam import focus_field_beam
    from biobeam.core.focus_field_lattice import focus_field_lattice

    return (
        Path,
        dedent,
        focus_field_beam,
        focus_field_lattice,
        mo,
        np,
        plt,
        re,
        tifffile,
    )


@app.cell
def _(Path):
    SETTINGS_PATH = Path(
        r"d:\codes\microscopy_proj\microscopy_data\New_lattice_sheet_data\20220329-30_CD63_deskew[433GB]\cell11\cell11_Settings.txt"
    )
    ORIGINAL_SCRIPT_PATH = Path(
        r"d:\codes\biobeam\bessel_lattice_lightsheet_psf_visualization_plotly可视化LLSM激发检测光.py"
    )
    OLD_PSF_PATH = Path(
        r"C:\Code\biobeam\psfdata\bessel_lattice_lightsheet_effective_psf.tif"
    )
    SAMPLE_PSF488_PATH = Path(
        r"C:\Code\biobeam\psfdata\20220329_488_square_0p55-0p50.tif"
    )
    GENERATED_PSF_PATH = Path(
        r"C:\Code\biobeam\psfdata\cell11_current_biobeam_effective_psf.tif"
    )
    return (
        GENERATED_PSF_PATH,
        OLD_PSF_PATH,
        ORIGINAL_SCRIPT_PATH,
        SAMPLE_PSF488_PATH,
        SETTINGS_PATH,
    )


@app.cell
def _(SETTINGS_PATH, re):
    settings_text = SETTINGS_PATH.read_text(encoding="utf-8", errors="replace") if SETTINGS_PATH.exists() else ""

    def _float_after_equals(label, default):
        _match = re.search(rf"{re.escape(label)}\s*=\s*([-+0-9.]+)", settings_text)
        return float(_match.group(1)) if _match else default

    def _bool_after_equals(label, default=False):
        _match = re.search(rf"{re.escape(label)}\s*=\s*(TRUE|FALSE|True|False)", settings_text)
        return (_match.group(1).lower() == "true") if _match else default

    def _line_after_colon(label):
        _match = re.search(rf"{re.escape(label)}\s*:\s*(.+)", settings_text)
        return _match.group(1).strip() if _match else ""

    _laser_line = _line_after_colon("Excitation Filter, Laser, Power (%), Exp(ms) (0)")
    _laser_values = [float(_v) for _v in re.findall(r"[-+]?\d+(?:\.\d+)?", _laser_line)]
    _sample_stack_line = _line_after_colon("S PZT Offset, Interval (um), # of Pixels for Excitation (0)")
    _sample_stack_values = [float(_v) for _v in re.findall(r"[-+]?\d+(?:\.\d+)?", _sample_stack_line)]

    cell11 = {
        "laser_nm": _laser_values[0] if _laser_values else 488.0,
        "laser_power_percent": _laser_values[1] if len(_laser_values) > 1 else float("nan"),
        "exposure_ms": _laser_values[2] if len(_laser_values) > 2 else float("nan"),
        "annular_inner_na": _float_after_equals("innerNA", 0.50),
        "annular_outer_na": _float_after_equals("outerNA", 0.55),
        "excitation_objective_na": _float_after_equals("Numerical Aperature", 0.80),
        "detection_na": 1.10,
        "emission_nm": 525.0,
        "medium_ri": 1.33,
        "sample_stack_offset_um": _sample_stack_values[0] if len(_sample_stack_values) > 0 else 50.0,
        "sample_stack_interval_um": _sample_stack_values[1] if len(_sample_stack_values) > 1 else 0.4,
        "sample_stack_pixels": int(_sample_stack_values[2]) if len(_sample_stack_values) > 2 else 201,
        "doe_installed": _bool_after_equals("DOE installed?", False),
        "doe_n_beams": int(_float_after_equals("# of beams", 10)),
        "doe_beam_spacing_um": _float_after_equals("Beam spacing (um)", 3.0),
        "stage_angle_deg": _float_after_equals("Angle between stage and bessel beam (deg)", 31.1089),
    }
    return (cell11,)


@app.cell(hide_code=True)
def _(OLD_PSF_PATH, ORIGINAL_SCRIPT_PATH, SETTINGS_PATH, cell11, dedent, mo):
    mo.md(
        dedent(
            f"""
            # 04: 用 cell11 参数重跑原始 BioBeam 脚本，并和旧 PSF 比较

            这个 notebook 的目标很窄：**不使用 03 的 LLSPY/SLM bridge**，而是把 cell11 的参数套进原始 BioBeam 模拟脚本的流程。

            原始流程保持为：

            `focus_field_lattice -> dither -> focus_field_beam -> np.swapaxes(excitation, 0, 1) -> excitation * detection`

            文件：

            - 原始脚本：`{ORIGINAL_SCRIPT_PATH}`
            - cell11 参数：`{SETTINGS_PATH}`
            - 旧 PSF：`{OLD_PSF_PATH}`

            cell11 读出的关键默认值：

            - excitation wavelength: `{cell11["laser_nm"]:.0f} nm`
            - annular NA: inner `{cell11["annular_inner_na"]:.3f}`, outer `{cell11["annular_outer_na"]:.3f}`
            - excitation objective NA cap: `{cell11["excitation_objective_na"]:.3f}`
            - detection NA: `{cell11["detection_na"]:.2f}`
            - sample stack: offset `{cell11["sample_stack_offset_um"]:.1f} um`, interval `{cell11["sample_stack_interval_um"]:.3f} um`, pixels `{cell11["sample_stack_pixels"]}`
            - DOE metadata: installed `{cell11["doe_installed"]}`, beams `{cell11["doe_n_beams"]}`, spacing `{cell11["doe_beam_spacing_um"]:.3f} um`

            注意：原始 BioBeam 脚本里面的 `kpoints` 不是 LLSPY 文件里的 `Beam spacing (um)`。这里默认使用 `kpoints=4` 来表示 square lattice，同时提供一个开关可以把 cell11 的 `# of beams=10` 暂时作为 `kpoints` 试算。
            """
        ).strip()
    )
    return


@app.cell
def _(OLD_PSF_PATH, SAMPLE_PSF488_PATH, np, tifffile):
    old_psf_raw = tifffile.imread(OLD_PSF_PATH).astype(np.float32)
    _sample_psf488_path = SAMPLE_PSF488_PATH if SAMPLE_PSF488_PATH.exists() else OLD_PSF_PATH
    sample_psf488_raw = tifffile.imread(_sample_psf488_path).astype(np.float32)
    return old_psf_raw, sample_psf488_raw


@app.cell
def _(cell11, mo, old_psf_raw):
    shape_n = mo.ui.slider(32, 128, step=2, value=int(old_psf_raw.shape[0]), label="shape N", show_value=True)
    voxel_um = mo.ui.slider(0.05, 0.30, step=0.01, value=0.10, label="voxel size um", show_value=True)

    exc_wavelength_nm = mo.ui.slider(405, 640, step=1, value=int(round(cell11["laser_nm"])), label="excitation wavelength nm", show_value=True)
    inner_na = mo.ui.slider(0.10, 0.95, step=0.005, value=float(cell11["annular_inner_na"]), label="NA1 inner", show_value=True)
    outer_na = mo.ui.slider(0.11, 1.20, step=0.005, value=float(cell11["annular_outer_na"]), label="NA2 outer", show_value=True)
    exc_obj_na = mo.ui.slider(0.30, 1.20, step=0.01, value=float(cell11["excitation_objective_na"]), label="excitation objective NA cap", show_value=True)
    sigma = mo.ui.slider(0.01, 0.50, step=0.01, value=0.30, label="sigma", show_value=True)
    kpoints = mo.ui.slider(3, 24, step=1, value=4, label="BioBeam kpoints (4=square)", show_value=True)
    use_cell11_beams_as_kpoints = mo.ui.checkbox(value=False, label="use cell11 # beams as kpoints")

    det_wavelength_nm = mo.ui.slider(500, 700, step=1, value=int(round(cell11["emission_nm"])), label="detection wavelength nm", show_value=True)
    det_na = mo.ui.slider(0.30, 1.20, step=0.01, value=float(cell11["detection_na"]), label="detection NA", show_value=True)
    medium_ri = mo.ui.slider(1.00, 1.52, step=0.01, value=float(cell11["medium_ri"]), label="medium RI", show_value=True)

    do_dither = mo.ui.checkbox(value=True, label="dither excitation")
    auto_dither_step = mo.ui.checkbox(value=True, label="use original-script dither step rule")
    dither_step = mo.ui.slider(1, 8, step=1, value=2, label="manual dither step pixels", show_value=True)
    align_peaks = mo.ui.checkbox(value=True, label="center peaks before comparison")
    norm_mode = mo.ui.dropdown(options=["max", "sum"], value="max", label="normalization")

    controls = mo.vstack(
        [
            mo.md("## 参数"),
            mo.hstack(
                [
                    mo.vstack([shape_n, voxel_um]),
                    mo.vstack([exc_wavelength_nm, inner_na, outer_na, exc_obj_na]),
                    mo.vstack([sigma, kpoints, use_cell11_beams_as_kpoints]),
                    mo.vstack([det_wavelength_nm, det_na, medium_ri]),
                    mo.vstack([do_dither, auto_dither_step, dither_step, align_peaks, norm_mode]),
                ]
            ),
        ]
    )
    return (
        align_peaks,
        auto_dither_step,
        controls,
        det_na,
        det_wavelength_nm,
        dither_step,
        do_dither,
        exc_obj_na,
        exc_wavelength_nm,
        inner_na,
        kpoints,
        medium_ri,
        norm_mode,
        outer_na,
        shape_n,
        sigma,
        use_cell11_beams_as_kpoints,
        voxel_um,
    )


@app.cell
def _(
    auto_dither_step,
    cell11,
    det_na,
    det_wavelength_nm,
    dither_step,
    do_dither,
    exc_obj_na,
    exc_wavelength_nm,
    focus_field_beam,
    focus_field_lattice,
    inner_na,
    kpoints,
    medium_ri,
    mo,
    np,
    outer_na,
    shape_n,
    sigma,
    use_cell11_beams_as_kpoints,
    voxel_um,
):
    sim_shape_n = int(shape_n.value)
    if sim_shape_n % 2:
        sim_shape_n += 1
    sim_shape = (sim_shape_n, sim_shape_n, sim_shape_n)
    sim_units = (float(voxel_um.value),) * 3

    lam_excitation = float(exc_wavelength_nm.value) / 1000.0
    lam_detection = float(det_wavelength_nm.value) / 1000.0
    NA1_excitation = float(inner_na.value)
    NA2_excitation = min(float(outer_na.value), float(exc_obj_na.value))
    NA2_excitation = max(NA2_excitation, NA1_excitation + 0.005)
    kpoints_excitation = int(cell11["doe_n_beams"]) if use_cell11_beams_as_kpoints.value else int(kpoints.value)
    n0 = float(medium_ri.value)

    with mo.status.spinner("Running original BioBeam workflow with cell11 defaults..."):
        psf_excitation_raw = focus_field_lattice(
            shape=sim_shape,
            units=sim_units,
            lam=lam_excitation,
            NA1=NA1_excitation,
            NA2=NA2_excitation,
            sigma=float(sigma.value),
            kpoints=kpoints_excitation,
            n0=n0,
            return_all_fields=False,
            n_integration_steps=200,
        ).astype(np.float32)

        if do_dither.value:
            if auto_dither_step.value:
                _max_na = max(NA1_excitation, NA2_excitation)
                _dx_parallel = lam_excitation / _max_na
                _dx_parallel_pixels = _dx_parallel / sim_units[0]
                used_dither_step = max(1, min(2, int(np.round(_dx_parallel_pixels))))
            else:
                used_dither_step = int(dither_step.value)

            _scan_positions = list(range(0, sim_shape_n, used_dither_step))
            psf_excitation_lightsheet = np.zeros_like(psf_excitation_raw, dtype=np.float32)
            for _pos in _scan_positions:
                psf_excitation_lightsheet += np.roll(psf_excitation_raw, _pos, axis=2)
            psf_excitation_lightsheet /= max(len(_scan_positions), 1)
        else:
            used_dither_step = 0
            _scan_positions = [0]
            psf_excitation_lightsheet = psf_excitation_raw

        psf_detection = focus_field_beam(
            shape=sim_shape,
            units=sim_units,
            lam=lam_detection,
            NA=float(det_na.value),
            n0=n0,
            return_all_fields=False,
            n_integration_steps=200,
        ).astype(np.float32)

    psf_excitation_det_coords = np.swapaxes(psf_excitation_lightsheet, 0, 1).astype(np.float32)
    psf_effective_cell11 = (psf_excitation_det_coords * psf_detection).astype(np.float32)

    sim_info = {
        "shape": sim_shape,
        "units_um": sim_units,
        "lam_excitation_um": lam_excitation,
        "NA1_excitation": NA1_excitation,
        "NA2_excitation_after_objective_cap": NA2_excitation,
        "excitation_objective_NA_cap": float(exc_obj_na.value),
        "sigma": float(sigma.value),
        "kpoints": kpoints_excitation,
        "lam_detection_um": lam_detection,
        "NA_detection": float(det_na.value),
        "n0": n0,
        "dither": bool(do_dither.value),
        "dither_step_pixels": used_dither_step,
        "scan_positions": len(_scan_positions),
    }
    return psf_effective_cell11, sim_info


@app.cell
def _(GENERATED_PSF_PATH, psf_effective_cell11, tifffile):
    GENERATED_PSF_PATH.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(GENERATED_PSF_PATH, psf_effective_cell11.astype("float32"), imagej=True)
    SAVED_PSF_PATH = GENERATED_PSF_PATH
    return (SAVED_PSF_PATH,)


@app.cell
def _(np):
    def normalize_volume(volume, mode="max"):
        _vol = np.asarray(volume, dtype=np.float32)
        _vol = np.nan_to_num(_vol, nan=0.0, posinf=0.0, neginf=0.0)
        _vol = np.maximum(_vol, 0.0)
        _den = float(np.sum(_vol)) if mode == "sum" else float(np.max(_vol))
        return (_vol / _den).astype(np.float32) if _den > 0 else np.zeros_like(_vol, dtype=np.float32)

    def center_peak(volume):
        _vol = np.asarray(volume, dtype=np.float32)
        if _vol.size == 0 or float(np.max(_vol)) <= 0:
            return _vol
        _peak = np.array(np.unravel_index(np.argmax(_vol), _vol.shape))
        _center = np.array(_vol.shape) // 2
        return np.roll(_vol, tuple(int(_v) for _v in (_center - _peak)), axis=(0, 1, 2))

    def center_crop_to_shape(volume, target_shape):
        _vol = np.asarray(volume, dtype=np.float32)
        _slices = []
        for _source, _target in zip(_vol.shape, target_shape):
            _width = min(int(_source), int(_target))
            _start = max((int(_source) - _width) // 2, 0)
            _slices.append(slice(_start, _start + _width))
        return _vol[tuple(_slices)].astype(np.float32)

    def center_crop_pair(generated, reference):
        _target_shape = tuple(min(int(_g), int(_r)) for _g, _r in zip(generated.shape, reference.shape))
        return center_crop_to_shape(generated, _target_shape), center_crop_to_shape(reference, _target_shape)

    def fwhm_pixels(profile):
        _p = np.asarray(profile, dtype=np.float64)
        if _p.size == 0 or float(np.max(_p)) <= 0:
            return float("nan")
        _half = 0.5 * float(np.max(_p))
        _idx = np.flatnonzero(_p >= _half)
        return float(_idx[-1] - _idx[0]) if _idx.size >= 2 else 0.0

    return center_crop_pair, center_peak, fwhm_pixels, normalize_volume


@app.cell
def _(
    align_peaks,
    center_crop_pair,
    center_peak,
    norm_mode,
    normalize_volume,
    old_psf_raw,
    psf_effective_cell11,
    sample_psf488_raw,
):
    cell11_norm = normalize_volume(psf_effective_cell11, norm_mode.value)
    old_norm = normalize_volume(old_psf_raw, norm_mode.value)
    sample488_norm = normalize_volume(sample_psf488_raw, norm_mode.value)
    if align_peaks.value:
        cell11_norm = center_peak(cell11_norm)
        old_norm = center_peak(old_norm)
        sample488_norm = center_peak(sample488_norm)
    cell11_matched, old_matched = center_crop_pair(cell11_norm, old_norm)
    cell11_matched_sample488, sample488_matched = center_crop_pair(cell11_norm, sample488_norm)
    old_matched_sample488_for_slices, sample488_matched_old_for_slices = center_crop_pair(old_norm, sample488_norm)
    diff = cell11_matched - old_matched
    abs_diff = abs(diff)
    diff_sample488 = cell11_matched_sample488 - sample488_matched
    abs_diff_sample488 = abs(diff_sample488)
    diff_old_sample488_for_slices = old_matched_sample488_for_slices - sample488_matched_old_for_slices
    abs_diff_old_sample488_for_slices = abs(diff_old_sample488_for_slices)
    return (
        abs_diff_old_sample488_for_slices,
        abs_diff_sample488,
        cell11_matched,
        cell11_matched_sample488,
        cell11_norm,
        diff,
        diff_old_sample488_for_slices,
        diff_sample488,
        old_matched,
        old_matched_sample488_for_slices,
        old_norm,
        sample488_matched,
        sample488_matched_old_for_slices,
        sample488_norm,
    )


@app.cell
def _(
    SAVED_PSF_PATH,
    cell11_matched,
    cell11_matched_sample488,
    fwhm_pixels,
    np,
    old_matched,
    old_psf_raw,
    psf_effective_cell11,
    sample488_matched,
    sample_psf488_raw,
    sim_info,
):
    def _metrics(a, b):
        _aa = np.ravel(a).astype(np.float64)
        _bb = np.ravel(b).astype(np.float64)
        _delta = _aa - _bb
        _corr = float(np.corrcoef(_aa, _bb)[0, 1]) if np.std(_aa) > 0 and np.std(_bb) > 0 else float("nan")
        _cos = float(np.dot(_aa, _bb) / (np.linalg.norm(_aa) * np.linalg.norm(_bb))) if np.linalg.norm(_aa) > 0 and np.linalg.norm(_bb) > 0 else float("nan")
        _rmse = float(np.sqrt(np.mean(_delta * _delta)))
        _mae = float(np.mean(np.abs(_delta)))
        _max_abs = float(np.max(np.abs(_delta)))
        _nrmse = _rmse / max(float(np.max(_bb) - np.min(_bb)), 1e-12)
        return _corr, _cos, _rmse, _nrmse, _mae, _max_abs

    def _rows_for_reference(reference_name, generated, reference, reference_raw):
        _corr, _cosine, _rmse, _nrmse, _mae, _max_abs = _metrics(generated, reference)
        _zc, _yc, _xc = [int(_s // 2) for _s in generated.shape]
        return [
            {"Reference": reference_name, "Metric": "Reference raw shape", "Value": str(tuple(int(_v) for _v in reference_raw.shape))},
            {"Reference": reference_name, "Metric": "Matched shape", "Value": str(tuple(int(_v) for _v in generated.shape))},
            {"Reference": reference_name, "Metric": "Pearson correlation", "Value": f"{_corr:.6f}"},
            {"Reference": reference_name, "Metric": "Cosine similarity", "Value": f"{_cosine:.6f}"},
            {"Reference": reference_name, "Metric": "RMSE", "Value": f"{_rmse:.6g}"},
            {"Reference": reference_name, "Metric": "NRMSE", "Value": f"{_nrmse:.6g}"},
            {"Reference": reference_name, "Metric": "MAE", "Value": f"{_mae:.6g}"},
            {"Reference": reference_name, "Metric": "Max abs diff", "Value": f"{_max_abs:.6g}"},
            {"Reference": reference_name, "Metric": "Generated FWHM z/y/x pixels", "Value": f"{fwhm_pixels(generated[:, _yc, _xc]):.2f} / {fwhm_pixels(generated[_zc, :, _xc]):.2f} / {fwhm_pixels(generated[_zc, _yc, :]):.2f}"},
            {"Reference": reference_name, "Metric": "Reference FWHM z/y/x pixels", "Value": f"{fwhm_pixels(reference[:, _yc, _xc]):.2f} / {fwhm_pixels(reference[_zc, :, _xc]):.2f} / {fwhm_pixels(reference[_zc, _yc, :]):.2f}"},
        ]

    metrics_table = [
        {"Reference": "simulation", "Metric": "Generated raw shape", "Value": str(tuple(int(_v) for _v in psf_effective_cell11.shape))},
        {"Reference": "simulation", "Metric": "Saved generated PSF", "Value": str(SAVED_PSF_PATH)},
        {"Reference": "simulation", "Metric": "Simulation parameters", "Value": str(sim_info)},
        {"Reference": "parameter comparison", "Metric": "Old BioBeam NA1 / NA2 / kpoints", "Value": "0.440 / 0.580 / 6"},
        {
            "Reference": "parameter comparison",
            "Metric": "Current simulated NA1 / NA2 / kpoints",
            "Value": f"{sim_info['NA1_excitation']:.3f} / {sim_info['NA2_excitation_after_objective_cap']:.3f} / {sim_info['kpoints']}",
        },
        {
            "Reference": "parameter comparison",
            "Metric": "Current lattice geometry",
            "Value": "square lattice" if int(sim_info["kpoints"]) == 4 else f"kpoints={sim_info['kpoints']}",
        },
    ]
    metrics_table.extend(_rows_for_reference("old BioBeam PSF", cell11_matched, old_matched, old_psf_raw))
    metrics_table.extend(_rows_for_reference("measured LLSM PSF", cell11_matched_sample488, sample488_matched, sample_psf488_raw))
    return (metrics_table,)


@app.cell
def _(
    abs_diff_old_sample488_for_slices,
    diff_old_sample488_for_slices,
    np,
    old_matched_sample488_for_slices,
    plt,
    sample488_matched_old_for_slices,
):
    _zc, _yc, _xc = [int(_s // 2) for _s in old_matched_sample488_for_slices.shape]
    _panels = [
        ("XY center z", old_matched_sample488_for_slices[_zc], sample488_matched_old_for_slices[_zc], diff_old_sample488_for_slices[_zc], abs_diff_old_sample488_for_slices[_zc]),
        ("XZ center y", old_matched_sample488_for_slices[:, _yc, :], sample488_matched_old_for_slices[:, _yc, :], diff_old_sample488_for_slices[:, _yc, :], abs_diff_old_sample488_for_slices[:, _yc, :]),
        ("YZ center x", old_matched_sample488_for_slices[:, :, _xc], sample488_matched_old_for_slices[:, :, _xc], diff_old_sample488_for_slices[:, :, _xc], abs_diff_old_sample488_for_slices[:, :, _xc]),
    ]
    fig_slices, _axes = plt.subplots(3, 4, figsize=(17, 12))
    _columns = ["old simulated PSF", "measured LLSM PSF", "signed diff", "abs diff"]
    _vmax = max(float(np.max(old_matched_sample488_for_slices)), float(np.max(sample488_matched_old_for_slices)), 1e-9)
    _diff_lim = max(abs(float(np.min(diff_old_sample488_for_slices))), abs(float(np.max(diff_old_sample488_for_slices))), 1e-9)
    _abs_vmax = max(float(np.max(abs_diff_old_sample488_for_slices)), 1e-9)
    for _row, (_title, _gen_slice, _old_slice, _diff_slice, _abs_slice) in enumerate(_panels):
        for _col, _image in enumerate([_gen_slice, _old_slice, _diff_slice, _abs_slice]):
            _ax = _axes[_row, _col]
            if _col == 2:
                _im = _ax.imshow(_image, origin="lower", cmap="coolwarm", vmin=-_diff_lim, vmax=_diff_lim, interpolation="nearest")
            elif _col == 3:
                _im = _ax.imshow(_image, origin="lower", cmap="magma", vmin=0, vmax=_abs_vmax, interpolation="nearest")
            else:
                _im = _ax.imshow(_image, origin="lower", cmap="magma", vmin=0, vmax=_vmax, interpolation="nearest")
            if _row == 0:
                _ax.set_title(_columns[_col])
            if _col == 0:
                _ax.set_ylabel(_title)
            _ax.set_xticks([])
            _ax.set_yticks([])
            fig_slices.colorbar(_im, ax=_ax, fraction=0.046, pad=0.03)
    fig_slices.tight_layout()
    return (fig_slices,)


@app.cell
def _(cell11_matched, old_matched, plt):
    _zc, _yc, _xc = [int(_s // 2) for _s in cell11_matched.shape]
    _profiles = [
        ("z profile", cell11_matched[:, _yc, _xc], old_matched[:, _yc, _xc]),
        ("y profile", cell11_matched[_zc, :, _xc], old_matched[_zc, :, _xc]),
        ("x profile", cell11_matched[_zc, _yc, :], old_matched[_zc, _yc, :]),
    ]
    fig_profiles, _axes = plt.subplots(1, 3, figsize=(15, 4))
    for _ax, (_title, _gen, _old) in zip(_axes, _profiles):
        _ax.plot(_gen, label="cell11 generated", linewidth=2)
        _ax.plot(_old, label="old PSF", linewidth=2, linestyle="--")
        _ax.set_title(_title)
        _ax.set_xlabel("pixel")
        _ax.set_ylabel("normalized intensity")
        _ax.grid(alpha=0.25)
        _ax.legend(fontsize=8)
    fig_profiles.tight_layout()
    return (fig_profiles,)


@app.cell
def _(cell11_matched, diff, old_matched, plt):
    _gen_flat = cell11_matched.ravel()
    _old_flat = old_matched.ravel()
    _diff_flat = diff.ravel()
    _step = max(1, _gen_flat.size // 25000)
    fig_distribution, _axes = plt.subplots(1, 2, figsize=(13, 5))
    _axes[0].scatter(_old_flat[::_step], _gen_flat[::_step], s=3, alpha=0.25)
    _lim = max(float(_old_flat.max()), float(_gen_flat.max()), 1e-9)
    _axes[0].plot([0, _lim], [0, _lim], color="black", linewidth=1)
    _axes[0].set_title("voxel scatter")
    _axes[0].set_xlabel("Old PSF")
    _axes[0].set_ylabel("cell11 generated")
    _axes[0].grid(alpha=0.25)
    _axes[1].hist(_diff_flat, bins=80, color="#5b7c99", alpha=0.85)
    _axes[1].set_title("signed difference histogram")
    _axes[1].set_xlabel("cell11 generated - old PSF")
    _axes[1].set_ylabel("voxel count")
    _axes[1].grid(alpha=0.25)
    fig_distribution.tight_layout()
    return (fig_distribution,)


@app.cell
def _(
    abs_diff_sample488,
    cell11_matched_sample488,
    diff_sample488,
    np,
    plt,
    sample488_matched,
):
    _zc, _yc, _xc = [int(_s // 2) for _s in cell11_matched_sample488.shape]
    _panels = [
        ("XY center z", cell11_matched_sample488[_zc], sample488_matched[_zc], diff_sample488[_zc], abs_diff_sample488[_zc]),
        ("XZ center y", cell11_matched_sample488[:, _yc, :], sample488_matched[:, _yc, :], diff_sample488[:, _yc, :], abs_diff_sample488[:, _yc, :]),
        ("YZ center x", cell11_matched_sample488[:, :, _xc], sample488_matched[:, :, _xc], diff_sample488[:, :, _xc], abs_diff_sample488[:, :, _xc]),
    ]
    fig_slices_sample488, _axes = plt.subplots(3, 4, figsize=(17, 12))
    _columns = ["New PSF based on cell11 parameter", "Measured PSF", "signed diff", "abs diff"]
    _vmax = max(float(np.max(cell11_matched_sample488)), float(np.max(sample488_matched)), 1e-9)
    _diff_lim = max(abs(float(np.min(diff_sample488))), abs(float(np.max(diff_sample488))), 1e-9)
    _abs_vmax = max(float(np.max(abs_diff_sample488)), 1e-9)
    for _row, (_title, _gen_slice, _sample_slice, _diff_slice, _abs_slice) in enumerate(_panels):
        for _col, _image in enumerate([_gen_slice, _sample_slice, _diff_slice, _abs_slice]):
            _ax = _axes[_row, _col]
            if _col == 2:
                _im = _ax.imshow(_image, origin="lower", cmap="coolwarm", vmin=-_diff_lim, vmax=_diff_lim, interpolation="nearest")
            elif _col == 3:
                _im = _ax.imshow(_image, origin="lower", cmap="magma", vmin=0, vmax=_abs_vmax, interpolation="nearest")
            else:
                _im = _ax.imshow(_image, origin="lower", cmap="magma", vmin=0, vmax=_vmax, interpolation="nearest")
            if _row == 0:
                _ax.set_title(_columns[_col])
            if _col == 0:
                _ax.set_ylabel(_title)
            _ax.set_xticks([])
            _ax.set_yticks([])
            fig_slices_sample488.colorbar(_im, ax=_ax, fraction=0.046, pad=0.03)
    fig_slices_sample488.tight_layout()
    return (fig_slices_sample488,)


@app.cell
def _(cell11_matched_sample488, plt, sample488_matched):
    _zc, _yc, _xc = [int(_s // 2) for _s in cell11_matched_sample488.shape]
    _profiles = [
        ("z profile", cell11_matched_sample488[:, _yc, _xc], sample488_matched[:, _yc, _xc]),
        ("y profile", cell11_matched_sample488[_zc, :, _xc], sample488_matched[_zc, :, _xc]),
        ("x profile", cell11_matched_sample488[_zc, _yc, :], sample488_matched[_zc, _yc, :]),
    ]
    fig_profiles_sample488, _axes = plt.subplots(1, 3, figsize=(15, 4))
    for _ax, (_title, _gen, _sample) in zip(_axes, _profiles):
        _ax.plot(_gen, label="cell11 generated", linewidth=2)
        _ax.plot(_sample, label="measured LLSM PSF", linewidth=2, linestyle="--")
        _ax.set_title(_title)
        _ax.set_xlabel("pixel")
        _ax.set_ylabel("normalized intensity")
        _ax.grid(alpha=0.25)
        _ax.legend(fontsize=8)
    fig_profiles_sample488.tight_layout()
    return (fig_profiles_sample488,)


@app.cell
def _(cell11_matched_sample488, diff_sample488, plt, sample488_matched):
    _gen_flat = cell11_matched_sample488.ravel()
    _sample_flat = sample488_matched.ravel()
    _diff_flat = diff_sample488.ravel()
    _step = max(1, _gen_flat.size // 25000)
    fig_distribution_sample488, _axes = plt.subplots(1, 2, figsize=(13, 5))
    _axes[0].scatter(_sample_flat[::_step], _gen_flat[::_step], s=3, alpha=0.25)
    _lim = max(float(_sample_flat.max()), float(_gen_flat.max()), 1e-9)
    _axes[0].plot([0, _lim], [0, _lim], color="black", linewidth=1)
    _axes[0].set_title("voxel scatter")
    _axes[0].set_xlabel("measured LLSM PSF")
    _axes[0].set_ylabel("cell11 generated")
    _axes[0].grid(alpha=0.25)
    _axes[1].hist(_diff_flat, bins=80, color="#7a6f9b", alpha=0.85)
    _axes[1].set_title("signed difference histogram")
    _axes[1].set_xlabel("cell11 generated - measured LLSM PSF")
    _axes[1].set_ylabel("voxel count")
    _axes[1].grid(alpha=0.25)
    fig_distribution_sample488.tight_layout()
    return (fig_distribution_sample488,)


@app.cell
def _(cell11_norm, old_norm, plt, sample488_norm):
    _raw_volumes = [
        ("old simulated PSF", old_norm),
        ("current simulated PSF", cell11_norm),
        ("measured LLSM PSF", sample488_norm),
    ]
    _target_shape = tuple(
        min(int(_vol.shape[_axis]) for _, _vol in _raw_volumes)
        for _axis in range(3)
    )

    def _center_crop_to_shape(_vol, _shape):
        _slices = []
        for _source, _target in zip(_vol.shape, _shape):
            _start = max((int(_source) - int(_target)) // 2, 0)
            _slices.append(slice(_start, _start + int(_target)))
        return _vol[tuple(_slices)]

    _volumes = [
        (_name, _center_crop_to_shape(_vol, _target_shape))
        for _name, _vol in _raw_volumes
    ]
    _planes = [
        ("XY center", lambda _v, _z, _y, _x: _v[_z, :, :]),
        ("XZ center", lambda _v, _z, _y, _x: _v[:, _y, :]),
        ("YZ center", lambda _v, _z, _y, _x: _v[:, :, _x]),
    ]
    fig_three_way_psf, _axes = plt.subplots(3, 3, figsize=(12, 10))
    for _col, (_name, _vol) in enumerate(_volumes):
        _zc, _yc, _xc = [int(_s // 2) for _s in _vol.shape]
        for _row, (_plane_name, _extract) in enumerate(_planes):
            _ax = _axes[_row, _col]
            _ax.imshow(_extract(_vol, _zc, _yc, _xc), origin="lower", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            if _row == 0:
                _ax.set_title(f"{_name}\ncenter-cropped {tuple(int(_s) for _s in _vol.shape)}")
            if _col == 0:
                _ax.set_ylabel(_plane_name)
            _ax.set_xticks([])
            _ax.set_yticks([])
    fig_three_way_psf.tight_layout()
    return (fig_three_way_psf,)


@app.cell(hide_code=True)
def _(
    controls,
    fig_distribution,
    fig_distribution_sample488,
    fig_profiles,
    fig_profiles_sample488,
    fig_slices,
    fig_slices_sample488,
    fig_three_way_psf,
    metrics_table,
    mo,
):
    mo.vstack(
        [
            controls,
            mo.md("## Numerical comparison"),
            mo.ui.table(metrics_table),
            mo.md("## Old simulated PSF vs measured LLSM PSF: center slices"),
            fig_slices,
            mo.md("## Old BioBeam PSF: center profiles"),
            fig_profiles,
            mo.md("## Old BioBeam PSF: voxel distribution"),
            fig_distribution,
            mo.md("## Measured LLSM PSF: center slices"),
            fig_slices_sample488,
            mo.md("## Measured LLSM PSF: center profiles"),
            fig_profiles_sample488,
            mo.md("## Measured LLSM PSF: voxel distribution"),
            fig_distribution_sample488,
            mo.md("## Three-way PSF image comparison"),
            fig_three_way_psf,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
