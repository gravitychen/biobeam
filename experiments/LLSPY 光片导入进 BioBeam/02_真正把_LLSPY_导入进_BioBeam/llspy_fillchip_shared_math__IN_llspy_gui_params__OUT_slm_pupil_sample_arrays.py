"""
用途: 共享 LLSPY Fill Chip 参数解析、SLM/pupil/sample 数组生成、CPU 光片传播和 Effective PSF 计算。
输入: LLSPY GUI 参数字典或 FillChipParams；可选本地 C:/Code/llspy-slm 原始项目。
输出: SLM binary mask、pupil after mask、sample intensity、CPU illumination volume、detection PSF、effective PSF。
用法: python llspy_fillchip_shared_math__IN_llspy_gui_params__OUT_slm_pupil_sample_arrays.py
依赖: numpy；可选 matplotlib；可选 C:/Code/llspy-slm 的 scipy/numba/Pillow 依赖用于原始 LLSPY 精确路径。
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


DEFAULT_LLSPY_REPO = Path(r"C:\Code\llspy-slm")


@dataclass
class FillChipParams:
    """参数名尽量贴近 LLSPY GUI，但统一使用微米和弧度。"""

    wavelength_um: float = 0.488
    inner_na: float = 0.500
    outer_na: float = 0.550
    shift_x_um: float = 0.0
    shift_y_um: float = 0.0
    tilt_rad: float = 0.0
    mag: float = 167.364
    crop: float = 0.220
    auto_spacing: bool = True
    spacing_um: Optional[float] = None
    fudge: float = 0.970
    auto_fill: bool = True
    n_beams: int = 53
    fillchip: float = 0.950
    slm_pixel_size_um: float = 13.662
    slm_xpix: int = 1280
    slm_ypix: int = 1024


def normalize01(array: np.ndarray) -> np.ndarray:
    """把数组线性归一化到 0..1，便于不同步骤直接对比。"""

    arr = np.asarray(array)
    if arr.size == 0:
        return arr.astype(np.float32)
    arr = arr.astype(np.float64, copy=False)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def log01(array: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """用于显示衍射图样的 log 版本，不改变原始数据。"""

    arr = normalize01(array).astype(np.float64)
    return np.log10(np.maximum(arr, floor))


def coerce_params(params: Any) -> FillChipParams:
    if isinstance(params, FillChipParams):
        return params
    if isinstance(params, dict):
        valid = {field.name for field in FillChipParams.__dataclass_fields__.values()}
        return FillChipParams(**{k: v for k, v in params.items() if k in valid})
    raise TypeError("params must be FillChipParams or dict")


def resolve_fill_chip_params(params: Any) -> Dict[str, Any]:
    """把 GUI 参数解析成 LLSPY 实际使用的派生参数。"""

    p = coerce_params(params)
    if p.wavelength_um <= 0:
        raise ValueError("wavelength_um must be > 0")
    if p.inner_na <= 0:
        raise ValueError("inner_na must be > 0")
    if p.outer_na <= p.inner_na:
        raise ValueError("outer_na must be larger than inner_na")
    if p.mag <= 0:
        raise ValueError("mag must be > 0")
    if p.slm_pixel_size_um <= 0:
        raise ValueError("slm_pixel_size_um must be > 0")
    if p.slm_xpix <= 16 or p.slm_ypix <= 16:
        raise ValueError("SLM pixel counts are too small")

    out = asdict(p)
    spacing_um = p.spacing_um
    if p.auto_spacing or spacing_um is None or spacing_um <= 0:
        spacing_um = p.fudge * p.wavelength_um / p.inner_na

    projected_pixel_um = p.slm_pixel_size_um / p.mag
    projected_width_um = p.slm_xpix * projected_pixel_um
    projected_half_width_um = projected_width_um / 2.0
    if p.auto_fill:
        n_beams = int(
            np.floor(1.0 + (p.fillchip * projected_half_width_um / spacing_um))
        )
    else:
        n_beams = int(p.n_beams)
    n_beams = max(n_beams, 1)

    out.update(
        {
            "spacing_um": float(spacing_um),
            "n_beams": int(n_beams),
            "total_phase_terms": int(2 * n_beams - 1),
            "projected_pixel_um": float(projected_pixel_um),
            "projected_width_um": float(projected_width_um),
            "projected_half_width_um": float(projected_half_width_um),
            "used_half_width_um": float(p.fillchip * projected_half_width_um),
            "annulus_inner_rad_per_um": float(p.inner_na * 2 * np.pi / p.wavelength_um),
            "annulus_outer_rad_per_um": float(p.outer_na * 2 * np.pi / p.wavelength_um),
        }
    )
    return out


def _dirichlet_sum(phase_ramp: np.ndarray, n_beams: int) -> np.ndarray:
    """LLSPY 中 1 + 2*sum(cos(i*f)) 的闭式写法。"""

    if n_beams <= 1:
        return np.ones_like(phase_ramp, dtype=np.float64)
    denominator = np.sin(0.5 * phase_ramp)
    numerator = np.sin((n_beams - 0.5) * phase_ramp)
    near_zero = np.abs(denominator) < 1e-8
    out = np.empty_like(phase_ramp, dtype=np.float64)
    out[near_zero] = 2 * n_beams - 1
    out[~near_zero] = numerator[~near_zero] / denominator[~near_zero]
    return out


def _center_crop(array: np.ndarray, rows: int, cols: int) -> np.ndarray:
    r0 = max((array.shape[0] - rows) // 2, 0)
    c0 = max((array.shape[1] - cols) // 2, 0)
    return array[r0 : r0 + rows, c0 : c0 + cols]


def simulate_llspy_like_numpy(
    params: Any,
    preview_pixels: int = 384,
) -> Dict[str, Any]:
    """
    快速 CPU 版 LLSPY Fourier 链路。

    这个函数保留 complex pupil field，所以 02 可以继续做 propagation。
    它不是逐像素复刻 GUI 的 1280x1024 原始函数，但物理步骤一致，适合交互预览。
    """

    resolved = resolve_fill_chip_params(params)
    nx = int(max(128, preview_pixels))
    if nx % 2:
        nx += 1
    ny = int(max(64, round(nx * resolved["slm_ypix"] / resolved["slm_xpix"])))
    if ny % 2:
        ny += 1

    # 保留真实 SLM 投影像素大小。预览降低 nx 时，相当于只看 SLM 中心区域；
    # 不能把整个芯片压缩进更少像素，否则 k-space 最大频率会降低，annular mask 会变空。
    dx_um = resolved["projected_pixel_um"]
    grid_n = nx + 1

    x_um = np.arange(-nx / 2, (nx + 1) / 2, 1.0, dtype=np.float64) * dx_um
    dk = 2 * np.pi / (nx + 1) / dx_um
    k_axis = np.arange(-nx / 2, (nx + 1) / 2, 1.0, dtype=np.float64) * dk
    kx, ky = np.meshgrid(k_axis, k_axis)
    kr = np.sqrt(kx * kx + ky * ky)

    inner_k = resolved["annulus_inner_rad_per_um"]
    outer_k = resolved["annulus_outer_rad_per_um"]
    pupil_mask = (kr < outer_k) & (kr > inner_k)

    phase_ramp = (
        kx * resolved["spacing_um"] * math.cos(resolved["tilt_rad"])
        + ky * resolved["spacing_um"] * math.sin(resolved["tilt_rad"])
    )
    phase_terms = _dirichlet_sum(phase_ramp, resolved["n_beams"])
    pupil_field_ideal = pupil_mask.astype(np.complex128) * phase_terms
    pupil_field_ideal *= np.exp(
        1j * (kx * resolved["shift_x_um"] + ky * resolved["shift_y_um"])
    )

    slm_field_ideal = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(pupil_field_ideal))
    ).real
    denom = np.max(np.abs(slm_field_ideal))
    if denom > 0:
        slm_field_ideal = slm_field_ideal / denom

    cropped_real = slm_field_ideal.copy()
    cropped_real[np.abs(cropped_real) < resolved["crop"]] = 0.0
    eps = np.finfo(float).eps
    slm_phase_full = np.sign(cropped_real + eps) * np.pi / 2.0 + np.pi / 2.0
    slm_phase_square = _center_crop(slm_phase_full, nx, nx)
    slm_binary = ((_center_crop(slm_phase_square, ny, nx) / np.pi) != 0).astype(
        np.uint8
    )

    # 这个 complex field 是 02 继续传播到 volume 的关键。
    slm_field = np.exp(1j * slm_phase_full)
    pupil_field_impinging = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
    pupil_field_after_mask_complex = pupil_field_impinging * pupil_mask
    sample_field = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(pupil_field_after_mask_complex))
    )

    pupil_impinging_intensity = np.real(
        pupil_field_impinging * np.conj(pupil_field_impinging)
    )
    pupil_intensity = np.real(
        pupil_field_after_mask_complex * np.conj(pupil_field_after_mask_complex)
    )
    sample_intensity = np.real(sample_field * np.conj(sample_field))

    return {
        "source": "fast_numpy_llspy_like",
        "resolved_params": resolved,
        "slm_binary_mask": slm_binary,
        "slm_field_ideal": normalize01(np.abs(slm_field_ideal)),
        "pupil_plane_impinging_intensity": normalize01(pupil_impinging_intensity),
        "pupil_plane_intensity": normalize01(pupil_intensity),
        "sample_plane_intensity": normalize01(sample_intensity),
        "complex_pupil_field_after_mask": pupil_field_after_mask_complex.astype(
            np.complex64
        ),
        "pupil_mask": pupil_mask.astype(np.uint8),
        "kx_rad_per_um": kx.astype(np.float32),
        "ky_rad_per_um": ky.astype(np.float32),
        "x_um": x_um.astype(np.float32),
        "dx_um": float(dx_um),
        "preview_pixels": int(nx),
        "preview_slm_shape": (int(ny), int(nx)),
    }


def run_original_llspy(
    params: Any,
    llspy_repo: Path = DEFAULT_LLSPY_REPO,
) -> Dict[str, Any]:
    """调用本地 tlambert03/llspy-slm 的原始函数，得到 GUI 一致的三张核心图。"""

    resolved = resolve_fill_chip_params(params)
    repo = Path(llspy_repo)
    if not repo.exists():
        raise FileNotFoundError(f"LLSPY repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from slmgen.slm import linear_bessel_array

    spacing_arg = None if resolved["auto_spacing"] else resolved["spacing_um"]
    n_beam_arg: Any = "fill" if resolved["auto_fill"] else int(resolved["n_beams"])
    slm_binary, sample_intensity, pupil_intensity = linear_bessel_array(
        wave=resolved["wavelength_um"],
        NA_inner=resolved["inner_na"],
        NA_outer=resolved["outer_na"],
        spacing=spacing_arg,
        n_beam=n_beam_arg,
        crop=resolved["crop"],
        tilt=resolved["tilt_rad"],
        shift_x=resolved["shift_x_um"],
        shift_y=resolved["shift_y_um"],
        mag=resolved["mag"],
        pixel=resolved["slm_pixel_size_um"],
        slm_xpix=resolved["slm_xpix"],
        slm_ypix=resolved["slm_ypix"],
        fillchip=resolved["fillchip"],
        fudge=resolved["fudge"],
        show=False,
        outdir=None,
        pattern_only=False,
    )

    return {
        "source": "llspy_original_linear_bessel_array",
        "resolved_params": resolved,
        "slm_binary_mask": np.asarray(slm_binary, dtype=np.uint8),
        "pupil_plane_intensity": normalize01(pupil_intensity),
        "sample_plane_intensity": normalize01(sample_intensity),
        "complex_pupil_field_after_mask": None,
        "llspy_repo": str(repo),
    }


def compute_fill_chip_outputs(
    params: Any,
    engine: str = "fast_numpy_llspy_like",
    preview_pixels: int = 384,
    llspy_repo: Path = DEFAULT_LLSPY_REPO,
) -> Dict[str, Any]:
    """统一入口：交互预览默认用 fast；需要 GUI 一致图像时用 llspy_original。"""

    if engine == "llspy_original":
        try:
            return run_original_llspy(params, llspy_repo=llspy_repo)
        except Exception as exc:  # pragma: no cover - 交互环境兜底
            fallback = simulate_llspy_like_numpy(params, preview_pixels=preview_pixels)
            fallback["source"] = "fast_numpy_fallback_after_llspy_error"
            fallback["llspy_error"] = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            return fallback
    return simulate_llspy_like_numpy(params, preview_pixels=preview_pixels)


def propagate_complex_pupil_to_volume(
    complex_pupil: np.ndarray,
    kx_rad_per_um: np.ndarray,
    ky_rad_per_um: np.ndarray,
    z_um: np.ndarray,
    wavelength_um: float,
    medium_ri: float = 1.33,
) -> np.ndarray:
    """用角谱法把 complex pupil field 传播成 3D illumination intensity volume。"""

    k0 = 2 * np.pi / wavelength_um
    kz = np.sqrt(
        np.maximum((medium_ri * k0) ** 2 - kx_rad_per_um**2 - ky_rad_per_um**2, 0.0)
    )
    reference_k = medium_ri * k0
    volume = np.empty((len(z_um),) + complex_pupil.shape, dtype=np.float32)
    for iz, z in enumerate(z_um):
        propagated_pupil = complex_pupil * np.exp(1j * (kz - reference_k) * z)
        field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(propagated_pupil)))
        volume[iz] = np.abs(field) ** 2
    return normalize01(volume)


def dither_excitation_volume(
    volume: np.ndarray,
    axis: int = 2,
) -> np.ndarray:
    """Integrate a scanned excitation beam into a continuous dithered light sheet."""

    dithered = np.mean(np.asarray(volume, dtype=np.float32), axis=axis, keepdims=True)
    return normalize01(np.broadcast_to(dithered, volume.shape).copy())


def excitation_to_detection_coordinates(volume: np.ndarray) -> np.ndarray:
    """
    Convert excitation coordinates to detection-objective coordinates.

    Input axes are (z_excitation_propagation, y_transverse, x_transverse).
    Output axes are (z_detection, y_excitation_propagation, x_transverse).
    """

    return np.swapaxes(volume, 0, 1).astype(np.float32, copy=False)


def make_gaussian_detection_psf(
    shape: Iterable[int],
    x_um: np.ndarray,
    z_um: np.ndarray,
    y_um: Optional[np.ndarray] = None,
    emission_wavelength_um: float = 0.525,
    detection_na: float = 0.80,
    medium_ri: float = 1.33,
) -> np.ndarray:
    """
    CPU fallback detection PSF。

    这里用高斯近似，不冒充 BioBeam/Gibson-Lanni 的完整检测 PSF；目的是让
    LLSPY illumination * detection PSF -> effective PSF 的数据流先跑通。
    """

    nz, ny, nx = [int(v) for v in shape]
    x = np.asarray(x_um, dtype=np.float64)
    if len(x) != nx:
        x = np.linspace(float(x.min()), float(x.max()), nx)
    if y_um is None:
        y = np.linspace(float(x.min()), float(x.max()), ny)
    else:
        y = np.asarray(y_um, dtype=np.float64)
        if len(y) != ny:
            y = np.linspace(float(y.min()), float(y.max()), ny)
    z = np.asarray(z_um, dtype=np.float64)
    if len(z) != nz:
        z = np.linspace(float(z.min()), float(z.max()), nz)

    fwhm_xy_um = 0.51 * emission_wavelength_um / max(detection_na, 1e-6)
    fwhm_z_um = 2.0 * medium_ri * emission_wavelength_um / max(detection_na**2, 1e-6)
    sigma_xy = max(fwhm_xy_um / 2.355, np.mean(np.diff(x)) / 2.0)
    sigma_z = max(fwhm_z_um / 2.355, np.mean(np.diff(z)) / 2.0)

    zz = z[:, None, None]
    yy = y[None, :, None]
    xx = x[None, None, :]
    psf = np.exp(-0.5 * ((xx / sigma_xy) ** 2 + (yy / sigma_xy) ** 2 + (zz / sigma_z) ** 2))
    return normalize01(psf)


def _center_crop_3d(array: np.ndarray, shape: Iterable[int]) -> np.ndarray:
    target = tuple(int(v) for v in shape)
    starts = [max((s - t) // 2, 0) for s, t in zip(array.shape, target)]
    slices = tuple(slice(start, start + size) for start, size in zip(starts, target))
    return array[slices]


def _mean_spacing_um(axis_um: np.ndarray) -> float:
    axis = np.asarray(axis_um, dtype=np.float64)
    if axis.size < 2:
        return 1.0
    return float(abs(np.mean(np.diff(axis))))


def make_biobeam_detection_psf(
    shape_zyx: Iterable[int],
    x_um: np.ndarray,
    y_um: np.ndarray,
    z_um: np.ndarray,
    emission_wavelength_um: float = 0.525,
    detection_na: float = 0.80,
    medium_ri: float = 1.33,
    n_integration_steps: int = 200,
) -> np.ndarray:
    """Use BioBeam's vectorial Debye PSF and return axes as (z, y, x)."""

    nz, ny, nx = [int(v) for v in shape_zyx]
    even_nx = nx + nx % 2
    even_ny = ny + ny % 2
    even_nz = nz + nz % 2
    from biobeam.core.focus_field_beam import focus_field_beam

    psf = focus_field_beam(
        shape=(even_nx, even_ny, even_nz),
        units=(
            _mean_spacing_um(x_um),
            _mean_spacing_um(y_um),
            _mean_spacing_um(z_um),
        ),
        lam=emission_wavelength_um,
        NA=detection_na,
        n0=medium_ri,
        return_all_fields=False,
        n_integration_steps=n_integration_steps,
    )
    psf = _center_crop_3d(np.asarray(psf), (nz, ny, nx))
    return normalize01(psf)


def make_detection_psf(
    shape_zyx: Iterable[int],
    x_um: np.ndarray,
    y_um: np.ndarray,
    z_um: np.ndarray,
    emission_wavelength_um: float = 0.525,
    detection_na: float = 0.80,
    medium_ri: float = 1.33,
    prefer_biobeam: bool = True,
) -> tuple[np.ndarray, str, Optional[str]]:
    if prefer_biobeam:
        try:
            return (
                make_biobeam_detection_psf(
                    shape_zyx,
                    x_um,
                    y_um,
                    z_um,
                    emission_wavelength_um=emission_wavelength_um,
                    detection_na=detection_na,
                    medium_ri=medium_ri,
                ),
                "biobeam_focus_field_beam",
                None,
            )
        except Exception as exc:  # pragma: no cover - depends on local OpenCL
            fallback = make_gaussian_detection_psf(
                shape_zyx,
                x_um,
                z_um,
                y_um=y_um,
                emission_wavelength_um=emission_wavelength_um,
                detection_na=detection_na,
                medium_ri=medium_ri,
            )
            return (
                fallback,
                "gaussian_fallback_after_biobeam_error",
                "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
    return (
        make_gaussian_detection_psf(
            shape_zyx,
            x_um,
            z_um,
            y_um=y_um,
            emission_wavelength_um=emission_wavelength_um,
            detection_na=detection_na,
            medium_ri=medium_ri,
        ),
        "gaussian_fallback",
        None,
    )


def compute_biobeam_bridge_outputs(
    params: Any,
    preview_pixels: int = 384,
    z_half_um: float = 12.0,
    nz: int = 81,
    medium_ri: float = 1.33,
    emission_wavelength_um: float = 0.525,
    detection_na: float = 0.80,
    dither_excitation: bool = True,
    prefer_biobeam_detection: bool = True,
) -> Dict[str, Any]:
    """
    02 的主流程：LLSPY-like complex pupil -> CPU illumination volume -> detection -> effective PSF。

    真实 BioBeam/GPU 可用时，illumination_volume_cpu 这一步可以替换成 BioBeam BPM。
    """

    llspy_like = simulate_llspy_like_numpy(params, preview_pixels=preview_pixels)
    z_um = np.linspace(-z_half_um, z_half_um, int(max(9, nz)), dtype=np.float32)
    illumination_raw = propagate_complex_pupil_to_volume(
        llspy_like["complex_pupil_field_after_mask"],
        llspy_like["kx_rad_per_um"],
        llspy_like["ky_rad_per_um"],
        z_um,
        llspy_like["resolved_params"]["wavelength_um"],
        medium_ri=medium_ri,
    )
    illumination_dithered = (
        dither_excitation_volume(illumination_raw, axis=2)
        if dither_excitation
        else illumination_raw
    )
    illumination = excitation_to_detection_coordinates(illumination_dithered)
    detection_z_um = llspy_like["x_um"]
    detection_y_um = z_um
    detection_x_um = llspy_like["x_um"]
    detection, detection_model, detection_error = make_detection_psf(
        illumination.shape,
        detection_x_um,
        detection_y_um,
        detection_z_um,
        emission_wavelength_um=emission_wavelength_um,
        detection_na=detection_na,
        medium_ri=medium_ri,
        prefer_biobeam=prefer_biobeam_detection,
    )
    effective = normalize01(illumination * detection)
    return {
        "source": "llspy_complex_pupil_to_cpu_biobeam_bridge",
        "resolved_params": llspy_like["resolved_params"],
        "llspy_like": llspy_like,
        "z_um": detection_z_um,
        "y_um": detection_y_um,
        "x_um": detection_x_um,
        "excitation_z_um": z_um,
        "illumination_volume_cpu": illumination,
        "illumination_volume_before_dither_cpu": illumination_raw,
        "illumination_volume_dithered_excitation_coords_cpu": illumination_dithered,
        "detection_psf_cpu": detection,
        "detection_psf_gaussian_cpu": detection,
        "effective_psf_cpu": effective,
        "medium_ri": float(medium_ri),
        "emission_wavelength_um": float(emission_wavelength_um),
        "detection_na": float(detection_na),
        "dither_excitation": bool(dither_excitation),
        "detection_model": detection_model,
        "detection_error": detection_error,
        "coordinate_system": "detection_objective_zyx",
    }


def array_stats(name: str, array: Any, physical_meaning: str) -> Dict[str, Any]:
    arr = np.asarray(array)
    if arr.size:
        values = np.abs(arr) if np.iscomplexobj(arr) else np.real(arr)
        amin = float(np.nanmin(values))
        amax = float(np.nanmax(values))
    else:
        amin = float("nan")
        amax = float("nan")
    return {
        "step": name,
        "shape": tuple(int(v) for v in arr.shape),
        "dtype": str(arr.dtype),
        "range": f"{amin:.4g} ~ {amax:.4g}",
        "physical_meaning": physical_meaning,
    }


def stats_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| 步骤名 | data shape | dtype | 数值范围 min~max | 每一步代码的物理意义 |",
        "|---|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {step} | `{shape}` | `{dtype}` | `{range}` | {physical_meaning} |".format(
                **row
            )
        )
    return "\n".join(lines)


def params_slug(resolved: Dict[str, Any]) -> str:
    return (
        "llspy_fillchip"
        f"__lam{resolved['wavelength_um'] * 1000:.0f}nm"
        f"__na{resolved['inner_na']:.3f}_{resolved['outer_na']:.3f}"
        f"__n{resolved['n_beams']:03d}"
        f"__sp{resolved['spacing_um']:.3f}"
        f"__crop{resolved['crop']:.3f}"
    ).replace(".", "p")


def save_bridge_outputs(output_root: Path, bridge: Dict[str, Any]) -> Path:
    outdir = Path(output_root) / params_slug(bridge["resolved_params"])
    outdir.mkdir(parents=True, exist_ok=True)
    ll = bridge["llspy_like"]
    np.savez_compressed(
        outdir / "arrays__llspy_to_biobeam_bridge.npz",
        slm_binary_mask=ll["slm_binary_mask"],
        pupil_plane_impinging_intensity=ll.get("pupil_plane_impinging_intensity"),
        pupil_plane_intensity=ll["pupil_plane_intensity"],
        sample_plane_intensity=ll["sample_plane_intensity"],
        illumination_volume_cpu=bridge["illumination_volume_cpu"],
        illumination_volume_before_dither_cpu=bridge["illumination_volume_before_dither_cpu"],
        illumination_volume_dithered_excitation_coords_cpu=bridge[
            "illumination_volume_dithered_excitation_coords_cpu"
        ],
        detection_psf_cpu=bridge["detection_psf_cpu"],
        detection_psf_gaussian_cpu=bridge["detection_psf_cpu"],
        effective_psf_cpu=bridge["effective_psf_cpu"],
        x_um=bridge["x_um"],
        y_um=bridge["y_um"],
        z_um=bridge["z_um"],
        excitation_z_um=bridge["excitation_z_um"],
    )
    summary = {
        "source": bridge["source"],
        "resolved_params": bridge["resolved_params"],
        "medium_ri": bridge["medium_ri"],
        "emission_wavelength_um": bridge["emission_wavelength_um"],
        "detection_na": bridge["detection_na"],
        "dither_excitation": bridge["dither_excitation"],
        "detection_model": bridge["detection_model"],
        "detection_error": bridge["detection_error"],
        "coordinate_system": bridge["coordinate_system"],
        "array_file": "arrays__llspy_to_biobeam_bridge.npz",
    }
    (outdir / "summary__llspy_to_biobeam_bridge.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (outdir / "README__what_is_inside.md").write_text(
        "\n".join(
            [
                "# LLSPY 光片导入 BioBeam 的中间结果",
                "",
                "- `slm_binary_mask`: LLSPY 的 binary SLM mask。",
                "- `pupil_plane_intensity`: 经过 annular mask 后的 pupil plane intensity。",
                "- `sample_plane_intensity`: LLSPY Fourier 链路得到的 sample plane intensity。",
                "- `illumination_volume_cpu`: dither 后并转换到 detection objective 坐标系的 illumination volume。",
                "- `detection_psf_cpu`: detection PSF；优先来自 BioBeam，失败时退回高斯近似。",
                "- `detection_psf_gaussian_cpu`: 兼容旧脚本的别名，内容等同于 `detection_psf_cpu`。",
                "- `effective_psf_cpu`: `illumination_volume_cpu * detection_psf_cpu`。",
                "- 坐标系: `(z_detection, y_excitation_propagation, x)`。",
            ]
        ),
        encoding="utf-8",
    )
    return outdir


if __name__ == "__main__":
    params = FillChipParams()
    outputs = compute_biobeam_bridge_outputs(params, preview_pixels=256, nz=31)
    rows = [
        array_stats(
            "slm_binary_mask",
            outputs["llspy_like"]["slm_binary_mask"],
            "SLM 上实际写入的 0/pi 二值相位图。",
        ),
        array_stats(
            "pupil_plane_intensity",
            outputs["llspy_like"]["pupil_plane_intensity"],
            "SLM 相位经过傅立叶变换和 annular mask 后的 pupil plane 强度。",
        ),
        array_stats(
            "illumination_volume_cpu",
            outputs["illumination_volume_cpu"],
            "complex pupil field 角谱传播得到的 3D excitation light sheet。",
        ),
        array_stats(
            "effective_psf_cpu",
            outputs["effective_psf_cpu"],
            "illumination 和 detection PSF 相乘后的 effective PSF。",
        ),
    ]
    print(stats_markdown(rows))
