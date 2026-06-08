#!/usr/bin/env python
# coding: utf-8
"""
Vectorial vs Scalar PSF Comparison
===================================
Compares BioBeam vectorial Debye-Wolf PSF (Richards-Wolf 1959 + Foreman & Toeroek 2011)
with Gohlke scalar Richards-Wolf PSF (isotropic scalar diffraction theory).

For each of Widefield and Confocal:
  - Orthogonal cross-sections: Vectorial | Scalar | Difference map
  - 1-D lateral (X, Y) and axial (Z) profiles with FWHM labels

Hardware parameters match the rest of the biobeam project.
"""

import sys
import os
import configparser

# Ensure biobeam repo root is on the path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Python 3.12+ compatibility patch for gputools
if sys.version_info >= (3, 12):
    if not hasattr(configparser, 'SafeConfigParser'):
        configparser.SafeConfigParser = configparser.ConfigParser

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psf as psflib

from biobeam.core.focus_field_beam import focus_field_beam

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Hardware parameters
# ---------------------------------------------------------------------------
NA     = 0.9
n0     = 1.33
lam_exc = 0.488   # μm
lam_em  = 0.525   # μm
dx = dy = dz = 0.1  # μm / voxel
SHAPE_VEC = (52, 52, 52)   # (Nx, Ny, Nz) — vectorial PSF shape

# Scalar PSF: shape=(27,27), dims=(2.6,2.6) → volume (53,53,53) at 0.1 μm/vx
# Crop [0:52, 0:52, 0:52] to match the vectorial grid exactly.
_SCALAR_NZR = 27
_SCALAR_DIM = 2.6   # μm (half-FOV)
_CROP = 52

# Ideal confocal pinhole (very small → equivalent to pointwise exc × em product)
PINHOLE_UM = 0.05   # μm ≈ 0.07 Airy units at detection wavelength

# ---------------------------------------------------------------------------
# Helper: FWHM
# ---------------------------------------------------------------------------
def compute_fwhm(profile, pixel_size):
    profile = np.asarray(profile, dtype=float)
    half = profile.max() / 2.0
    above = profile >= half
    idx = np.where(above)[0]
    if len(idx) < 2:
        return np.nan
    i0, i1 = idx[0], idx[-1]
    left  = (i0-1) + (half - profile[i0-1]) / (profile[i0] - profile[i0-1]) if i0 > 0 else float(i0)
    right = float(i1) + (half - profile[i1]) / (profile[i1+1] - profile[i1]) if i1 < len(profile)-1 else float(i1)
    return (right - left) * pixel_size


# ---------------------------------------------------------------------------
# 1. Vectorial PSFs (BioBeam)
# ---------------------------------------------------------------------------
print("=== Computing Vectorial PSFs (BioBeam) ===")
print("Detection PSF (Widefield) ...")
_vec_det = focus_field_beam(shape=SHAPE_VEC, units=(dx, dy, dz),
                            lam=lam_em, NA=NA, n0=n0,
                            n_integration_steps=200)

print("Excitation PSF ...")
_vec_exc = focus_field_beam(shape=SHAPE_VEC, units=(dx, dy, dz),
                            lam=lam_exc, NA=NA, n0=n0,
                            n_integration_steps=200)

psf_vec_wf = _vec_det / _vec_det.max()
_confocal_raw = _vec_exc * _vec_det
psf_vec_cf = _confocal_raw / _confocal_raw.max()

print(f"  Vectorial Widefield shape: {psf_vec_wf.shape}")
print(f"  Vectorial Confocal shape:  {psf_vec_cf.shape}")


# ---------------------------------------------------------------------------
# 2. Scalar PSFs (Gohlke psf library — isotropic scalar Richards-Wolf)
# ---------------------------------------------------------------------------
print("\n=== Computing Scalar PSFs (Gohlke isotropic) ===")

_sc_args = dict(
    shape=(_SCALAR_NZR, _SCALAR_NZR),
    dims=(_SCALAR_DIM, _SCALAR_DIM),
    num_aperture=NA,
    refr_index=n0,
)

print("Scalar Widefield (emission PSF) ...")
_sc_wf_obj = psflib.PSF(
    psflib.ISOTROPIC | psflib.EMISSION,
    em_wavelen=lam_em * 1e3,   # library takes nm
    **_sc_args,
)
_sc_wf_vol = _sc_wf_obj.volume()             # (53, 53, 53) at 0.1 μm/vx
psf_sc_wf = _sc_wf_vol[:_CROP, :_CROP, :_CROP].astype(np.float32)
psf_sc_wf = psf_sc_wf / psf_sc_wf.max()

print("Scalar Confocal (ideal point pinhole) ...")
_sc_cf_obj = psflib.PSF(
    psflib.ISOTROPIC | psflib.CONFOCAL,
    ex_wavelen=lam_exc * 1e3,
    em_wavelen=lam_em  * 1e3,
    pinhole_radius=PINHOLE_UM,
    pinhole_shape='round',
    **_sc_args,
)
_sc_cf_vol = _sc_cf_obj.volume()             # (53, 53, 53) at 0.1 μm/vx
psf_sc_cf = _sc_cf_vol[:_CROP, :_CROP, :_CROP].astype(np.float32)
psf_sc_cf = psf_sc_cf / psf_sc_cf.max()

print(f"  Scalar Widefield shape: {psf_sc_wf.shape}")
print(f"  Scalar Confocal shape:  {psf_sc_cf.shape}")


# ---------------------------------------------------------------------------
# 3. Coordinate axes
# ---------------------------------------------------------------------------
Nz, Ny, Nx = psf_vec_wf.shape
cz, cy, cx = Nz//2, Ny//2, Nx//2
x_coords = (np.arange(Nx) - cx) * dx
y_coords = (np.arange(Ny) - cy) * dy
z_coords = (np.arange(Nz) - cz) * dz


# ---------------------------------------------------------------------------
# 4. Difference maps (vectorial − scalar)
# ---------------------------------------------------------------------------
diff_wf = psf_vec_wf - psf_sc_wf
diff_cf = psf_vec_cf - psf_sc_cf


# ---------------------------------------------------------------------------
# 5. Plotting helpers
# ---------------------------------------------------------------------------
def centre_slices(vol):
    """Return XY, XZ, YZ centre slices from a (Nz, Ny, Nx) volume."""
    Nz, Ny, Nx = vol.shape
    cz, cy, cx = Nz//2, Ny//2, Nx//2
    xy = vol[cz, :, :]    # (Ny, Nx) — focal plane
    xz = vol[:, cy, :]    # (Nz, Nx) — axial through y=0
    yz = vol[:, :, cx]    # (Nz, Ny) — axial through x=0
    return xy, xz, yz

PLANE_LABELS = ['XY  (focal plane)', 'XZ  (axial)', 'YZ  (axial)']
EXTENTS = [
    [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
    [x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
    [y_coords[0], y_coords[-1], z_coords[0], z_coords[-1]],
]
XLABELS = ['X (μm)', 'X (μm)', 'Y (μm)']
YLABELS = ['Y (μm)', 'Z (μm)', 'Z (μm)']


def make_comparison_figure(psf_vec, psf_sc, diff, title, fname):
    """
    3-row × 3-col figure:
      Row 0: Vectorial PSF slices
      Row 1: Scalar PSF slices
      Row 2: Difference map (vectorial − scalar)
    Columns: XY | XZ | YZ
    """
    row_labels = ['Vectorial\n(BioBeam)', 'Scalar\n(Gohlke)', 'Difference\n(Vec − Sc)']
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(title, fontsize=12, y=0.99)

    abs_max_diff = np.abs(diff).max()
    vmax_diff = max(abs_max_diff, 1e-6)

    for col, (plane, ext, xl, yl) in enumerate(zip(PLANE_LABELS, EXTENTS, XLABELS, YLABELS)):
        xy_v, xz_v, yz_v = centre_slices(psf_vec)
        xy_s, xz_s, yz_s = centre_slices(psf_sc)
        xy_d, xz_d, yz_d = centre_slices(diff)

        slices_rows = [
            [xy_v, xz_v, yz_v],
            [xy_s, xz_s, yz_s],
            [xy_d, xz_d, yz_d],
        ]

        for row in range(3):
            ax = axes[row, col]
            sl = slices_rows[row][col]

            if row < 2:
                im = ax.imshow(sl, origin='lower', cmap='hot',
                               extent=ext, vmin=0, vmax=1, aspect='equal')
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Norm. intensity', fontsize=7)
            else:
                im = ax.imshow(sl, origin='lower', cmap='RdBu_r',
                               extent=ext, vmin=-vmax_diff, vmax=vmax_diff,
                               aspect='equal')
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(f'Δ (max={abs_max_diff:.3f})', fontsize=7)

            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            if col == 0:
                ax.set_title(f'{row_labels[row]}\n{plane}', fontsize=9)
            else:
                ax.set_title(plane, fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Profile comparison figure (lateral + axial, log scale)
# ---------------------------------------------------------------------------
def make_profile_figure(psf_vec_wf, psf_sc_wf, psf_vec_cf, psf_sc_cf):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f'Vectorial vs Scalar PSF Profiles (log scale)\n'
        f'NA={NA}, n₀={n0}, λ_exc={lam_exc} μm, λ_em={lam_em} μm',
        fontsize=12
    )

    datasets = [
        (psf_vec_wf, psf_sc_wf, 'Widefield'),
        (psf_vec_cf, psf_sc_cf, 'Confocal'),
    ]

    for col, (pvec, psc, label) in enumerate(datasets):
        # Lateral X profile
        px_v = pvec[cz, cy, :]
        px_s = psc[cz, cy, :]
        fwhm_v = compute_fwhm(px_v, dx)
        fwhm_s = compute_fwhm(px_s, dx)

        ax = axes[0, col]
        ax.semilogy(x_coords, px_v / px_v.max(), color='steelblue',
                    label=f'Vectorial  FWHM={fwhm_v:.3f} μm')
        ax.semilogy(x_coords, px_s / px_s.max(), color='tomato', linestyle='--',
                    label=f'Scalar     FWHM={fwhm_s:.3f} μm')
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax.set_title(f'{label} — Lateral X Profile')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Norm. intensity (log)')
        ax.set_ylim(1e-4, 2)
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)

        # Axial Z profile
        pz_v = pvec[:, cy, cx]
        pz_s = psc[:, cy, cx]
        fwhm_zv = compute_fwhm(pz_v, dz)
        fwhm_zs = compute_fwhm(pz_s, dz)

        ax = axes[1, col]
        ax.semilogy(z_coords, pz_v / pz_v.max(), color='steelblue',
                    label=f'Vectorial  FWHM={fwhm_zv:.3f} μm')
        ax.semilogy(z_coords, pz_s / pz_s.max(), color='tomato', linestyle='--',
                    label=f'Scalar     FWHM={fwhm_zs:.3f} μm')
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax.set_title(f'{label} — Axial Z Profile')
        ax.set_xlabel('Z (μm)')
        ax.set_ylabel('Norm. intensity (log)')
        ax.set_ylim(1e-4, 2)
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'compare_profiles.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Statistics summary
# ---------------------------------------------------------------------------
def print_stats(name, psf_vec, psf_sc, diff):
    px_v = psf_vec[cz, cy, :]
    px_s = psf_sc[cz, cy, :]
    pz_v = psf_vec[:, cy, cx]
    pz_s = psf_sc[:, cy, cx]

    fwhm_lat_v = compute_fwhm(px_v, dx)
    fwhm_lat_s = compute_fwhm(px_s, dx)
    fwhm_ax_v  = compute_fwhm(pz_v, dz)
    fwhm_ax_s  = compute_fwhm(pz_s, dz)

    print(f"\n--- {name} ---")
    print(f"  Lateral FWHM:  Vectorial={fwhm_lat_v:.3f} μm   Scalar={fwhm_lat_s:.3f} μm   "
          f"ratio={fwhm_lat_v/fwhm_lat_s:.4f}")
    print(f"  Axial FWHM:    Vectorial={fwhm_ax_v:.3f} μm   Scalar={fwhm_ax_s:.3f} μm   "
          f"ratio={fwhm_ax_v/fwhm_ax_s:.4f}")
    print(f"  Difference map: max={diff.max():.4f}  min={diff.min():.4f}  "
          f"RMSE={np.sqrt(np.mean(diff**2)):.4f}  "
          f"MAE={np.mean(np.abs(diff)):.4f}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("\n=== Generating Figures ===")

make_comparison_figure(
    psf_vec_wf, psf_sc_wf, diff_wf,
    title=(
        f'Widefield PSF: Vectorial (BioBeam) vs Scalar (Gohlke)\n'
        f'NA={NA}, n₀={n0}, λ_em={lam_em} μm, voxel={dx}×{dy}×{dz} μm'
    ),
    fname='compare_widefield.png'
)

make_comparison_figure(
    psf_vec_cf, psf_sc_cf, diff_cf,
    title=(
        f'Confocal PSF: Vectorial (BioBeam) vs Scalar (Gohlke)\n'
        f'NA={NA}, n₀={n0}, λ_exc={lam_exc} μm, λ_em={lam_em} μm, '
        f'pinhole={PINHOLE_UM} μm'
    ),
    fname='compare_confocal.png'
)

make_profile_figure(psf_vec_wf, psf_sc_wf, psf_vec_cf, psf_sc_cf)

print("\n=== FWHM & Difference Statistics ===")
print_stats('Widefield', psf_vec_wf, psf_sc_wf, diff_wf)
print_stats('Confocal',  psf_vec_cf, psf_sc_cf, diff_cf)

print("\nDone.")
