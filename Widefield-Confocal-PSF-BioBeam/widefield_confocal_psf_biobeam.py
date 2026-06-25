#!/usr/bin/env python
# coding: utf-8
"""
Widefield and Confocal PSF via BioBeam Vectorial Debye-Wolf Integration

Uses focus_field_beam() (Richards-Wolf 1959 + Foreman & Toeroek 2011) to compute:
  - Widefield PSF  = PSF_detection(lam=0.525, NA=1.10)
  - Confocal PSF   = PSF_excitation(lam=0.488, NA=1.10) x PSF_detection(lam=0.525, NA=1.10)

Hardware parameters consistent with the rest of the biobeam project.
"""

import sys
import os
import configparser

# Ensure biobeam repo root is on the path when running from subdirectory
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
from tifffile import imwrite

from biobeam.core.focus_field_beam import focus_field_beam

# ---------------------------------------------------------------------------
# Hardware parameters
# ---------------------------------------------------------------------------
shape  = (52, 52, 52)        # (Nx, Ny, Nz) — even values required by focus_field_beam
units  = (0.1, 0.1, 0.1)    # (dx, dy, dz) in microns => FOV ≈ 5.2 x 5.2 x 5.2 um
NA     = 1.10                # shared objective NA (excitation & detection)
n0     = 1.33                # water immersion
lam_exc = 0.488              # excitation wavelength (um)
lam_em  = 0.525              # emission / detection wavelength (um)
n_steps = 200                # Debye-Wolf integration steps

out_dir = os.path.dirname(os.path.abspath(__file__))
psfdata_dir = os.path.join(_repo_root, "psfdata")

# ---------------------------------------------------------------------------
# 1. Compute PSFs
# ---------------------------------------------------------------------------
print("Computing Detection PSF (Widefield) ...")
psf_det = focus_field_beam(
    shape=shape, units=units,
    lam=lam_em, NA=NA, n0=n0,
    return_all_fields=False,
    n_integration_steps=n_steps
)

print("Computing Excitation PSF ...")
psf_exc = focus_field_beam(
    shape=shape, units=units,
    lam=lam_exc, NA=NA, n0=n0,
    return_all_fields=False,
    n_integration_steps=n_steps
)

# Widefield = detection only (uniform illumination has no focal PSF)
psf_widefield = psf_det / psf_det.max()

# Confocal = excitation x detection (point illumination + pinhole detection)
psf_confocal_raw = psf_exc * psf_det
psf_confocal = psf_confocal_raw / psf_confocal_raw.max()

print(f"Widefield PSF  shape={psf_widefield.shape}  range=[{psf_widefield.min():.3f}, {psf_widefield.max():.3f}]")
print(f"Confocal PSF   shape={psf_confocal.shape}   range=[{psf_confocal.min():.3f}, {psf_confocal.max():.3f}]")

os.makedirs(psfdata_dir, exist_ok=True)
widefield_out = os.path.join(psfdata_dir, "widefield_effective.tif")
confocal_out = os.path.join(psfdata_dir, "confocal_psf.tif")
imwrite(widefield_out, psf_widefield.astype(np.float32), imagej=True)
imwrite(confocal_out, psf_confocal.astype(np.float32), imagej=True)
print(f"Saved widefield PSF: {widefield_out}")
print(f"Saved confocal PSF: {confocal_out}")

# ---------------------------------------------------------------------------
# 2. Helper: FWHM from a 1-D profile
# ---------------------------------------------------------------------------
def compute_fwhm(profile, pixel_size):
    """Return FWHM in microns via linear interpolation around the half-max."""
    profile = np.asarray(profile, dtype=float)
    half_max = profile.max() / 2.0
    above = profile >= half_max
    indices = np.where(above)[0]
    if len(indices) < 2:
        return np.nan
    # left edge: interpolate between indices[0]-1 and indices[0]
    i0 = indices[0]
    if i0 > 0:
        frac = (half_max - profile[i0 - 1]) / (profile[i0] - profile[i0 - 1])
        left = (i0 - 1) + frac
    else:
        left = float(i0)
    # right edge: interpolate between indices[-1] and indices[-1]+1
    i1 = indices[-1]
    if i1 < len(profile) - 1:
        frac = (half_max - profile[i1]) / (profile[i1 + 1] - profile[i1])
        right = float(i1) + frac
    else:
        right = float(i1)
    return (right - left) * pixel_size


# ---------------------------------------------------------------------------
# 3. Extract centre profiles
# ---------------------------------------------------------------------------
Nz, Ny, Nx = psf_widefield.shape   # focus_field_beam returns (Nz, Ny, Nx)
cz, cy, cx = Nz // 2, Ny // 2, Nx // 2
dx, dy, dz = units

x_coords = (np.arange(Nx) - cx) * dx
y_coords = (np.arange(Ny) - cy) * dy
z_coords = (np.arange(Nz) - cz) * dz

def profiles(psf):
    return (
        psf[cz, cy, :],   # x-profile (lateral)
        psf[cz, :, cx],   # y-profile (lateral)
        psf[:, cy, cx],   # z-profile (axial)
    )

wf_x, wf_y, wf_z = profiles(psf_widefield)
cf_x, cf_y, cf_z = profiles(psf_confocal)

# FWHM values
fwhm_wf_x = compute_fwhm(wf_x, dx)
fwhm_wf_y = compute_fwhm(wf_y, dy)
fwhm_wf_z = compute_fwhm(wf_z, dz)
fwhm_cf_x = compute_fwhm(cf_x, dx)
fwhm_cf_y = compute_fwhm(cf_y, dy)
fwhm_cf_z = compute_fwhm(cf_z, dz)

print("\n=== FWHM Summary ===")
print(f"Widefield  lateral X: {fwhm_wf_x:.3f} um   Y: {fwhm_wf_y:.3f} um   axial Z: {fwhm_wf_z:.3f} um")
print(f"Confocal   lateral X: {fwhm_cf_x:.3f} um   Y: {fwhm_cf_y:.3f} um   axial Z: {fwhm_cf_z:.3f} um")
if not np.isnan(fwhm_wf_x) and not np.isnan(fwhm_cf_x):
    print(f"Confocal/Widefield lateral ratio: {fwhm_cf_x / fwhm_wf_x:.3f}  (theory ≈ {1/np.sqrt(2):.3f})")

# ---------------------------------------------------------------------------
# 4. Figure 1 — Orthogonal cross-sections  (2 rows x 3 cols)
# ---------------------------------------------------------------------------
def centre_slices(psf):
    Nz, Ny, Nx = psf.shape
    cz, cy, cx = Nz // 2, Ny // 2, Nx // 2
    return psf[cz, :, :], psf[:, cy, :], psf[:, :, cx]   # XY, XZ, YZ

labels_col = ['XY (lateral)', 'XZ (axial)', 'YZ (axial)']
extents = [
    [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],   # XY
    [x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],   # XZ
    [y_coords[0], y_coords[-1], z_coords[0], z_coords[-1]],   # YZ
]
xlabels = ['X (μm)', 'X (μm)', 'Y (μm)']
ylabels = ['Y (μm)', 'Z (μm)', 'Z (μm)']

fig1, axes1 = plt.subplots(2, 3, figsize=(12, 8))
fig1.suptitle(
    f'BioBeam Vectorial PSF — Orthogonal Sections\n'
    f'NA={NA}, n0={n0}, λ_exc={lam_exc} μm, λ_em={lam_em} μm, voxel={dx}×{dy}×{dz} μm',
    fontsize=11
)

for col, (plane_label, ext, xl, yl) in enumerate(zip(labels_col, extents, xlabels, ylabels)):
    wf_slices = centre_slices(psf_widefield)
    cf_slices = centre_slices(psf_confocal)
    for row, (psf_slice, row_label, fwhm_xy) in enumerate([
        (wf_slices[col], 'Widefield', fwhm_wf_x),
        (cf_slices[col], 'Confocal',  fwhm_cf_x),
    ]):
        ax = axes1[row, col]
        im = ax.imshow(
            psf_slice, origin='lower', cmap='hot',
            extent=ext, vmin=0, vmax=1, aspect='equal'
        )
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        if col == 0:
            ax.set_title(f'{row_label}\n{plane_label}', fontsize=10)
        else:
            ax.set_title(plane_label, fontsize=10)
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Norm. intensity')

fig1.tight_layout()
out1 = os.path.join(out_dir, 'psf_orthogonal.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out1}")

# ---------------------------------------------------------------------------
# 5. Figure 2 — Lateral & axial profiles (log scale)
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle(
    f'BioBeam PSF Profiles (log scale) — NA={NA}, n0={n0}',
    fontsize=12
)

# Lateral (x-profile)
ax = axes2[0]
ax.semilogy(x_coords, wf_x / wf_x.max(), label=f'Widefield (FWHM={fwhm_wf_x:.3f} μm)', color='steelblue')
ax.semilogy(x_coords, cf_x / cf_x.max(), label=f'Confocal  (FWHM={fwhm_cf_x:.3f} μm)', color='tomato')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Half-max')
ax.set_xlabel('X (μm)')
ax.set_ylabel('Normalised intensity (log)')
ax.set_title('Lateral X Profile')
ax.set_ylim(1e-4, 2)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

# Axial (z-profile)
ax = axes2[1]
ax.semilogy(z_coords, wf_z / wf_z.max(), label=f'Widefield (FWHM={fwhm_wf_z:.3f} μm)', color='steelblue')
ax.semilogy(z_coords, cf_z / cf_z.max(), label=f'Confocal  (FWHM={fwhm_cf_z:.3f} μm)', color='tomato')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Half-max')
ax.set_xlabel('Z (μm)')
ax.set_ylabel('Normalised intensity (log)')
ax.set_title('Axial Z Profile')
ax.set_ylim(1e-4, 2)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

fig2.tight_layout()
out2 = os.path.join(out_dir, 'psf_profiles.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved: {out2}")

print("\nDone.")
