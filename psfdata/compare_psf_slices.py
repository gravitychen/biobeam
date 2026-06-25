#!/usr/bin/env python
# coding: utf-8

"""Compare all PSF TIFF files used in the BioBeam experiments.

The script normalizes each PSF, aligns its peak to the volume center, center-crops
all volumes to the same shape, and saves a center-slice comparison figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "psf_slices_comparison.png"

PSF_FILES = {
    "cylindrical lightsheet": SCRIPT_DIR / "cylindrical_lightsheet_effective_psf.tif",
    "old bessel lattice": SCRIPT_DIR / "bessel_lattice_lightsheet_effective_psf.tif",
    "current cell11 lattice": SCRIPT_DIR / "cell11_current_biobeam_effective_psf.tif",
    "measured LLSM": SCRIPT_DIR / "20220329_488_square_0p55-0p50.tif",
    "confocal": SCRIPT_DIR / "confocal_psf.tif",
    "widefield": SCRIPT_DIR / "widefield_effective.tif",
}


def normalize_volume(volume):
    vol = np.asarray(volume, dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol = np.maximum(vol, 0.0)
    vmax = float(np.max(vol))
    return vol / vmax if vmax > 0 else np.zeros_like(vol, dtype=np.float32)


def center_peak(volume):
    vol = np.asarray(volume, dtype=np.float32)
    if vol.size == 0 or float(np.max(vol)) <= 0:
        return vol
    peak = np.array(np.unravel_index(np.argmax(vol), vol.shape))
    center = np.array(vol.shape) // 2
    return np.roll(vol, tuple(int(v) for v in center - peak), axis=(0, 1, 2))


def center_crop_to_shape(volume, target_shape):
    slices = []
    for source, target in zip(volume.shape, target_shape):
        width = min(int(source), int(target))
        start = max((int(source) - width) // 2, 0)
        slices.append(slice(start, start + width))
    return volume[tuple(slices)]


def load_psfs():
    psfs = {}
    metadata = {}
    for name, path in PSF_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
        raw = imread(path)
        psfs[name] = center_peak(normalize_volume(raw))
        metadata[name] = {
            "filename": path.name,
            "raw_shape": tuple(int(v) for v in raw.shape),
            "dtype": str(raw.dtype),
        }
        print(f"{name}: {path.name}, raw shape={tuple(raw.shape)}, dtype={raw.dtype}")
    return psfs, metadata


def crop_all_to_common_shape(psfs):
    target_shape = tuple(
        min(int(volume.shape[axis]) for volume in psfs.values())
        for axis in range(3)
    )
    cropped = {
        name: center_crop_to_shape(volume, target_shape)
        for name, volume in psfs.items()
    }
    print(f"Common center-cropped shape: {target_shape}")
    return cropped


def plot_center_slices(psfs, metadata):
    planes = [
        ("XY center", lambda vol, z, y, x: vol[z, :, :]),
        ("XZ center", lambda vol, z, y, x: vol[:, y, :]),
        ("YZ center", lambda vol, z, y, x: vol[:, :, x]),
    ]

    fig, axes = plt.subplots(3, len(psfs), figsize=(4.0 * len(psfs), 10), dpi=160)
    if len(psfs) == 1:
        axes = axes[:, np.newaxis]

    for col, (name, volume) in enumerate(psfs.items()):
        zc, yc, xc = [int(size // 2) for size in volume.shape]
        for row, (plane_name, extract) in enumerate(planes):
            ax = axes[row, col]
            ax.imshow(
                extract(volume, zc, yc, xc),
                origin="lower",
                cmap="magma",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            if row == 0:
                raw_shape = metadata[name]["raw_shape"]
                ax.set_title(
                    f"{name}\nraw {raw_shape}\ncrop {tuple(volume.shape)}",
                    fontsize=9,
                )
            if col == 0:
                ax.set_ylabel(plane_name, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    return fig


def main():
    psfs, metadata = load_psfs()
    psfs = crop_all_to_common_shape(psfs)
    fig = plot_center_slices(psfs, metadata)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
