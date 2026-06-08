import marimo

__generated_with = "0.21.1"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import special as spc
    from math import sqrt

    return mo, np, plt, spc, sqrt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interactive Pupil Function -> PSF

    Drag sliders to change pupil parameters. Three panels update live:

    - **Left**: Pupil P(kx, ky)
    - **Middle**: Focal plane |FT{P}|^2
    - **Right**: Beam propagation xz (angular spectrum)

    xz uses: `E(x,z) = iFFT{ P * exp(i*2*pi*kz*z) }` where `kz = sqrt(1 - kx^2 - ky^2)`.

    This is exactly what BioBeam `psf_debye.cl` does.
    """)
    return


@app.cell
def _(np):
    LAM_EXCITATION = 0.488
    N0 = 1.33
    N = 256
    NZ = 100
    Z_HALF = 15

    ks = np.linspace(-1.0, 1.0, N)
    KX, KY = np.meshgrid(ks, ks)
    KR = np.sqrt(KX**2 + KY**2)
    KZ = np.sqrt(np.maximum(1.0 - KR**2, 0.0))
    zs = np.linspace(-Z_HALF, Z_HALF, NZ)

    _xr = np.fft.fftshift(np.fft.fftfreq(N, d=2.0 / N))
    XR, _YR = np.meshgrid(_xr, _xr)

    BEAM_TYPES = [
        "DSLM (Gaussian)",
        "Bessel (ring)",
        "SPIM (cylindrical)",
        "Lattice LLS",
    ]
    return BEAM_TYPES, KR, KX, KY, KZ, LAM_EXCITATION, N, N0, NZ, XR, zs


@app.cell
def _(KR, KX, KY, KZ, N, NZ, np, zs):
    def make_pupil(beam_type, NA, NA1, NA2, kpoints, sigma_phi, line_hw):
        if beam_type == "DSLM (Gaussian)":
            return (KR <= NA).astype(complex)
        if beam_type == "Bessel (ring)":
            return ((KR >= NA1) & (KR <= NA2)).astype(complex)
        if beam_type == "SPIM (cylindrical)":
            return ((np.abs(KY) < line_hw) & (KR <= NA)).astype(complex)
        if beam_type == "Lattice LLS":
            # Match BioBeam focus_field_lattice: Gaussian blobs on an annular ring.
            k_center = (NA1 + NA2) / 2.0
            ring = (KR >= NA1) & (KR <= NA2)
            ts = np.pi * (0.5 + 2.0 / kpoints * np.arange(kpoints))
            kxs = k_center * np.cos(ts)
            kys = k_center * np.sin(ts)
            sigma_kx = k_center * 0.04
            sigma_ky = max(sigma_phi, 1e-4)

            amp = np.zeros((N, N))
            for kx_i, ky_i in zip(kxs, kys):
                amp += (
                    np.exp(-((KX - kx_i) ** 2) / (2 * sigma_kx**2))
                    * np.exp(-((KY - ky_i) ** 2) / (2 * sigma_ky**2))
                )
            amp *= ring
            if amp.max() > 0:
                amp /= amp.max()
            return amp.astype(complex)
        raise ValueError(f"Unknown beam type: {beam_type}")

    def focal_psf(P):
        E = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(P)))
        return np.abs(E) ** 2

    def xz_profile(P):
        xz = np.zeros((NZ, N), dtype=np.float32)
        for iz, z in enumerate(zs):
            Ep = P * np.exp(1j * 2 * np.pi * KZ * z)
            E = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ep)))
            xz[iz] = np.abs(E[N // 2, :]) ** 2
        return xz

    def log_norm(img, decades=3):
        eps = img.max() * 1e-10 + 1e-30
        lo = np.log10(img.max() + eps) - decades
        return np.clip(np.log10(img + eps), lo, None)

    return focal_psf, log_norm, make_pupil, xz_profile


@app.cell
def _(BEAM_TYPES, mo):
    w_type = mo.ui.dropdown(
        options=BEAM_TYPES,
        value="DSLM (Gaussian)",
        label="Beam",
    )
    w_NA = mo.ui.slider(
        0.05, 0.95, step=0.01, value=0.30, label="NA", show_value=True
    )
    w_NA1 = mo.ui.slider(
        0.05, 0.85, step=0.01, value=0.50, label="NA1 inner", show_value=True
    )
    w_NA2 = mo.ui.slider(
        0.10, 0.95, step=0.01, value=0.55, label="NA2 outer", show_value=True
    )
    w_kpts = mo.ui.slider(
        2, 100, step=1, value=4, label="kpoints", show_value=True
    )
    w_sigma = mo.ui.slider(
        0.02, 0.60, step=0.02, value=0.15, label="sigma_phi", show_value=True
    )
    w_linehw = mo.ui.slider(
        0.005, 0.10, step=0.005, value=0.015, label="line_width", show_value=True
    )
    return w_NA, w_NA1, w_NA2, w_kpts, w_linehw, w_sigma, w_type


@app.cell
def _(
    LAM_EXCITATION,
    N,
    N0,
    focal_psf,
    log_norm,
    make_pupil,
    mo,
    np,
    plt,
    w_NA,
    w_NA1,
    w_NA2,
    w_kpts,
    w_linehw,
    w_sigma,
    w_type,
    xz_profile,
):
    _P = make_pupil(
        w_type.value,
        w_NA.value,
        w_NA1.value,
        w_NA2.value,
        w_kpts.value,
        w_sigma.value,
        w_linehw.value,
    )
    _c, _CROP = N // 2, 80

    _xy = focal_psf(_P)[_c - _CROP : _c + _CROP, _c - _CROP : _c + _CROP]
    _lxy = log_norm(_xy)
    _xz = xz_profile(_P)[:, _c - _CROP : _c + _CROP]
    _lxz = log_norm(_xz)

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4))
    _fig.suptitle("Pupil Function and PSF", fontsize=12)

    _theta = np.linspace(0, 2 * np.pi, 300)
    _r_px = N // 2 - 2
    _axes[0].imshow(np.abs(_P), cmap="Reds", origin="lower", vmin=0, vmax=1)
    _axes[0].plot(
        N // 2 + _r_px * np.cos(_theta),
        N // 2 + _r_px * np.sin(_theta),
        "k--",
        lw=0.8,
        alpha=0.6,
    )
    _axes[0].set_title("Pupil P(kx,ky)", fontsize=10)
    _axes[0].axis("off")

    _axes[1].imshow(_lxy, cmap="inferno", origin="lower")
    _axes[1].set_title("Focal plane |FT{P}|^2", fontsize=10)
    _axes[1].axis("off")

    _axes[2].imshow(_lxz, cmap="inferno", origin="lower", aspect="auto")
    _axes[2].set_title("Propagation xz", fontsize=10)
    _axes[2].axis("off")

    _fig.tight_layout()

    mo.vstack(
        [
            mo.md("## Controls"),
            mo.md(
                f"Default LLSM parameters: lambda_exc={LAM_EXCITATION:.3f} um, "
                f"n0={N0:.2f}, NA_inner=0.50, NA_outer=0.55, kpoints=4 (square)."
            ),
            w_type,
            mo.hstack(
                [
                    mo.vstack([mo.md("**DSLM / SPIM**"), w_NA, w_linehw]),
                    mo.vstack([mo.md("**Bessel / Lattice**"), w_NA1, w_NA2]),
                    mo.vstack([mo.md("**Lattice only**"), w_kpts, w_sigma]),
                ]
            ),
            _fig,
        ]
    )

    # 参数总结：λ_exc=0.488 µm, NA_inner=0.44, NA_outer=0.58, sigma=0.1, kpoints=6, n0=1.33, voxel=0.1 µm, volume=52³。
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Suggested Experiments

    **Exp 1 - DSLM: effect of NA**
    Drag NA 0.1->0.9. Larger NA = tighter focus (xy), shorter depth of field (xz).

    **Exp 2 - Bessel: ring width**
    Fix NA1=0.50, drag NA2. Thinner ring = longer non-diffracting propagation, more side lobes.

    **Exp 3 - Lattice: kpoints**
    Default kpoints=4 gives a square lattice. Change kpoints 2->4->6 to compare line, square, and hex-like patterns.

    **Exp 4 - Lattice: sigma_phi**
    Fix kpoints=4, drag sigma_phi 0.02->0.50. Larger sigma = thinner sheet, shorter DOF.
    This is the core trade-off in Chen et al. 2014.

    **Exp 5 - SPIM vs DSLM**
    Same NA, switch beam type. SPIM xz = line focus (sheet directly). DSLM xz = point focus (needs scanning).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Comparison: BioBeam vs `llspy-slm` ([tlambert03/llspy-slm](https://github.com/tlambert03/llspy-slm))

    | Aspect | BioBeam (this notebook) | llspy-slm |
    |--------|------------------------|-----------|
    | **Lattice pupil model** | Gaussian blobs at polygon vertices on ring (sigma_phi tunable) | Plane-wave waveset -> real-space pattern -> binary 0/pi SLM phase -> FFT -> annular mask |
    | **Widefield PSF** | Ideal Airy disc, no aberrations (simple disk aperture) | Gibson-Lanni: Fourier-Bessel expansion of OPD (coverslip RI, immersion RI, particle depth) |
    | **Output** | Continuous complex pupil -> angular spectrum propagation | Binary SLM bitmap for hardware; optional intensity at sample |

    ### Lattice LLS approach

    **BioBeam** places `kpoints` Gaussian blobs at polygon vertices on the annular ring.
    `sigma_phi` controls the ky spread -> trade-off between sheet thickness and depth of field.

    **llspy-slm-style lattice approximation shown below**:
    1. Precomputed square waveset: 4 exact plane-wave k-directions
    2. Superpose plane waves in real space -> ideal 2D square lattice
    3. Apply Gaussian bounding envelope (width proportional to pi / (NA_outer - NA_inner))
    4. Binarize to 0/pi SLM phase -> FFT creates diffraction orders
    5. Annular mask selects the desired ring -> field at sample

    The binarization step is the key hardware-facing difference: llspy-slm generates actual SLM bitmap files.
    """)
    return


@app.cell
def _(
    KR,
    KX,
    KY,
    LAM_EXCITATION,
    N,
    N0,
    XR,
    focal_psf,
    log_norm,
    make_pupil,
    mo,
    np,
    plt,
    xz_profile,
):
    WAVESET_SQUARE = np.array(
        [
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    def make_pupil_llspy_ideal(NA1, NA2, sigma_k=0.012):
        k_c = (NA1 + NA2) / 2.0
        ring = (KR >= NA1) & (KR <= NA2)
        amp = np.zeros((N, N))
        for kxn, kyn in WAVESET_SQUARE:
            amp += np.exp(
                -((KX - k_c * kxn) ** 2 + (KY - k_c * kyn) ** 2)
                / (2 * sigma_k**2)
            )
        amp *= ring
        if amp.max() > 0:
            amp /= amp.max()
        return amp.astype(complex)

    def make_pupil_llspy_slm(NA1, NA2, fill_factor=0.75, crop=0.15):
        P_ideal = make_pupil_llspy_ideal(NA1, NA2)
        E_real = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(P_ideal)))
        real_e = np.real(E_real)
        kxdiff = max(NA2 - NA1, 0.01)
        sigma_env = (np.pi / kxdiff) / (
            np.sqrt(2.0 * np.log(2.0)) * fill_factor
        )
        real_e *= np.exp(-2.0 * (XR / sigma_env) ** 2)
        real_e /= np.abs(real_e).max() + 1e-30
        real_e[np.abs(real_e) < crop] = 0.0

        eps = np.finfo(float).eps
        slm_phase = np.sign(real_e + eps) * np.pi / 2 + np.pi / 2
        E_slm = np.exp(1j * slm_phase)
        E_pupil = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_slm)))
        ring = (KR >= NA1) & (KR <= NA2)
        E_pupil *= ring

        amp = np.abs(E_pupil)
        if amp.max() > 0:
            amp /= amp.max()
        return amp.astype(complex)

    _NA1_c, _NA2_c, _kpoints_c = 0.50, 0.55, 4

    with mo.status.spinner("Computing Lattice LLS pupils and PSFs (xz may take ~30 s)..."):
        _P_bb = make_pupil("Lattice LLS", None, _NA1_c, _NA2_c, _kpoints_c, 0.15, None)
        _P_ll_id = make_pupil_llspy_ideal(_NA1_c, _NA2_c)
        _P_ll_sl = make_pupil_llspy_slm(_NA1_c, _NA2_c)

        _pupils = [_P_bb, _P_ll_id, _P_ll_sl]
        _titles = [
            "BioBeam square\n(Gaussian blobs, sigma_phi=0.15)",
            "Square waveset ideal\n(delta-like functions)",
            "Square SLM binary\n(0/pi phase -> FFT -> annular mask)",
        ]
        _row_labels = [
            "Pupil P(kx, ky)",
            "Focal plane |FT{P}|^2 (log)",
            "xz propagation (log)",
        ]
        _c, _CRP = N // 2, 60

        _fig_ll, _ax_ll = plt.subplots(3, 3, figsize=(13, 10))
        _fig_ll.suptitle(
            f"Lattice LLS square default - lambda={LAM_EXCITATION:.3f} um, "
            f"n0={N0:.2f}, NA_inner={_NA1_c}, NA_outer={_NA2_c}, kpoints={_kpoints_c}",
            fontsize=12,
            y=1.01,
        )
        for _col, (_P, _ttl) in enumerate(zip(_pupils, _titles)):
            _ax_ll[0, _col].set_title(_ttl, fontsize=9)
            for _row in range(3):
                _ax_ll[_row, _col].axis("off")
                if _col == 0:
                    _ax_ll[_row, _col].set_ylabel(_row_labels[_row], fontsize=8)
            _ax_ll[0, _col].imshow(
                np.abs(_P), cmap="Reds", origin="lower", vmin=0, vmax=1
            )
            _psf = focal_psf(_P)[_c - _CRP : _c + _CRP, _c - _CRP : _c + _CRP]
            _ax_ll[1, _col].imshow(log_norm(_psf), cmap="inferno", origin="lower")
            _xz = xz_profile(_P)[:, _c - _CRP : _c + _CRP]
            _ax_ll[2, _col].imshow(
                log_norm(_xz), cmap="inferno", origin="lower", aspect="auto"
            )
        _fig_ll.tight_layout()
        _fig_ll.savefig("biobeam_vs_llspy_lattice.png", dpi=150, bbox_inches="tight")

    _fig_ll
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gibson-Lanni PSF (Widefield) - `llspy-slm/slmgen/makepsf.py`

    `makePSF()` uses a **Fourier-Bessel series expansion** of the total optical path difference (OPD):

    ```
    W(rho, z) = (2*pi/lambda) * [OPD_sample + OPD_immersion + OPD_coverslip]
    OPD_sample     = d_particle * sqrt(n_sample^2 - NA^2*rho^2)
    OPD_immersion  = (z + WD)*sqrt(n_i^2 - NA^2*rho^2) - WD*sqrt(n_i0^2 - NA^2*rho^2)
    OPD_coverslip  = t_g*sqrt(n_g^2 - NA^2*rho^2) - t_g0*sqrt(n_g0^2 - NA^2*rho^2)
    ```

    BioBeam's `(KR <= NA)` gives a perfect **Airy** PSF with no aberrations.
    `makePSF()` gives a realistic aberrated PSF when sample RI != immersion RI, or particle is deep in sample.
    """)
    return


@app.cell
def _(N, focal_psf, log_norm, make_pupil, mo, np, plt, spc, sqrt):
    def makePSF_llspy(
        wavelength=0.525,
        NA=0.9,
        nx=129,
        nz=129,
        dx=0.05,
        dz=0.05,
        RI=1.33,
        immRI=1.515,
        csRI=1.515,
        csthick=170,
        workingdistance=150,
        particledistance=0,
        num_basis=100,
        num_samples=500,
    ):
        ni = ni0 = immRI
        ng = ng0 = csRI
        tg = tg0 = csthick
        min_wl = 0.436
        sf = NA * (3 * np.arange(1, num_basis + 1) - 2) * min_wl / wavelength
        x0 = (nx - 1) / 2
        mr = round(sqrt(2) * (nx - x0)) + 1
        r = dx * np.arange(0, mr)
        a = min(NA, RI, ni, ni0, ng, ng0) / NA
        rho = np.linspace(0, a, num_samples)
        z = dz * np.arange(-nz / 2, nz / 2) + dz / 2
        NArho2 = NA**2 * rho**2

        OPDs = particledistance * np.sqrt(np.maximum(RI**2 - NArho2, 0))
        OPDi = (z[:, None] + workingdistance) * np.sqrt(
            np.maximum(ni**2 - NArho2, 0)
        ) - workingdistance * np.sqrt(np.maximum(ni0**2 - NArho2, 0))
        OPDg = tg * np.sqrt(np.maximum(ng**2 - NArho2, 0)) - tg0 * np.sqrt(
            np.maximum(ng0**2 - NArho2, 0)
        )

        W = 2 * np.pi / wavelength * (OPDs + OPDi + OPDg)
        phase = np.cos(W) + 1j * np.sin(W)
        J = spc.jv(0, sf[:, None] * rho)
        C, *_ = np.linalg.lstsq(J.T, phase.T, rcond=None)
        b = 2 * np.pi * r[:, None] * NA / wavelength
        denom = sf * sf - b * b
        R = (
            sf * spc.jv(1, sf * a) * spc.jv(0, b * a) * a
            - b * spc.jv(0, sf * a) * spc.jv(1, b * a) * a
        ) / denom
        PSF_rz = (np.abs(R.dot(C)) ** 2).T
        PSF_rz /= PSF_rz.max()
        return PSF_rz

    _kw = dict(
        wavelength=0.488,
        NA=0.8,
        nx=129,
        nz=129,
        dx=0.05,
        dz=0.05,
        RI=1.33,
        immRI=1.515,
        csRI=1.515,
        csthick=170,
        workingdistance=150,
    )

    with mo.status.spinner("Computing Gibson-Lanni PSFs..."):
        _GL_ideal = makePSF_llspy(**_kw, particledistance=0)
        _GL_aber = makePSF_llspy(**_kw, particledistance=15)
        _P_airy = make_pupil("DSLM (Gaussian)", _kw["NA"], None, None, None, None, None)
        _psf_bb2d = focal_psf(_P_airy)[N // 2 - 60 : N // 2 + 60, N // 2 - 60 : N // 2 + 60]

    _nz2 = _GL_ideal.shape[0] // 2
    _nr = _GL_ideal.shape[1]
    _r_um = np.arange(_nr) * _kw["dx"]

    _fig_gl, _ax_gl = plt.subplots(1, 3, figsize=(14, 4))
    _fig_gl.suptitle(
        f"Widefield PSF: BioBeam (Debye, ideal) vs llspy-slm Gibson-Lanni  NA={_kw['NA']}  lambda=488 nm",
        fontsize=11,
    )
    _ax_gl[0].imshow(log_norm(_psf_bb2d), cmap="inferno", origin="lower")
    _ax_gl[0].set_title("BioBeam focal plane (log)\nIdeal Airy, no aberrations", fontsize=9)
    _ax_gl[0].axis("off")
    _ax_gl[1].imshow(np.log10(_GL_ideal + 1e-6), cmap="inferno", origin="lower", aspect="auto")
    _ax_gl[1].set_title("llspy-slm GL xz (log)\nNo aberration (particle at coverslip)", fontsize=9)
    _ax_gl[1].axis("off")
    _ax_gl[2].imshow(np.log10(_GL_aber + 1e-6), cmap="inferno", origin="lower", aspect="auto")
    _ax_gl[2].set_title("llspy-slm GL xz (log)\n15 um into water (n=1.33), glass obj (n=1.515)", fontsize=9)
    _ax_gl[2].axis("off")
    _fig_gl.tight_layout()
    _fig_gl.savefig("biobeam_vs_llspy_widefield_xz.png", dpi=150, bbox_inches="tight")

    _fig_rp, _ax_rp = plt.subplots(figsize=(7, 4))
    _ax_rp.semilogy(
        _r_um,
        _GL_ideal[_nz2] / _GL_ideal[_nz2, 0],
        lw=2,
        label="GL focal, no aberration",
    )
    _ax_rp.semilogy(
        _r_um,
        _GL_aber[_nz2] / _GL_aber[_nz2, 0],
        lw=2,
        ls="--",
        label="GL focal, 15 um depth (water/glass RI mismatch)",
    )
    _psf_r_bb = _psf_bb2d[60, 60:] / _psf_bb2d[60, 60]
    _ax_rp.semilogy(
        np.linspace(0, _r_um[-1], len(_psf_r_bb)),
        _psf_r_bb,
        lw=2,
        ls=":",
        label="BioBeam ideal Airy (scale matched)",
    )
    _ax_rp.set_xlabel("r (um)")
    _ax_rp.set_ylabel("Intensity (log, normalized to peak)")
    _ax_rp.set_title(f"Focal-plane radial profiles  NA={_kw['NA']}  lambda=488 nm")
    _ax_rp.set_xlim(0, 2.0)
    _ax_rp.set_ylim(1e-4, 1.5)
    _ax_rp.legend(fontsize=8)
    _fig_rp.tight_layout()
    _fig_rp.savefig("biobeam_vs_llspy_widefield_profiles.png", dpi=150, bbox_inches="tight")

    mo.vstack([_fig_gl, _fig_rp])
    return


if __name__ == "__main__":
    app.run()
