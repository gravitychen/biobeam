# psf_example.py

"""Point Spread Function example.

Demonstrate the use of the psf library for calculating point spread functions
for fluorescence microscopy.

"""

from __future__ import annotations

from typing import Any

import numpy
import psf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def psf_example(
    cmap: str = 'hot',
    savebin: bool = False,
    savetif: bool = False,
    savevol: bool = False,
    plot: bool = True,
    **kwargs: Any,
) -> None:
    """Calculate, save, and plot various point spread functions."""
    args = {
        'shape': (512, 512),  # number of samples in z and r direction
        'dims': (5.0, 5.0),  # size in z and r direction in micrometers
        'ex_wavelen': 488.0,  # excitation wavelength in nanometers
        'em_wavelen': 520.0,  # emission wavelength in nanometers
        'num_aperture': 1.2,
        'refr_index': 1.333,
        'magnification': 1.0,
        'pinhole_radius': 0.05,  # in micrometers
        'pinhole_shape': 'square',
    }
    args.update(kwargs)

    obsvol = psf.PSF(
        psf.ISOTROPIC | psf.CONFOCAL, **args  # type: ignore[arg-type]
        # psf.ISOTROPIC | psf.WIDEFIELD, **args  # type: ignore[arg-type]
    )
    expsf = obsvol.expsf
    empsf = obsvol.empsf

    gauss = gauss2 = psf.PSF(
        psf.GAUSSIAN | psf.EXCITATION, **args  # type: ignore[arg-type]
    )

    assert expsf is not None
    assert empsf is not None

    print(expsf)
    print(empsf)
    print(obsvol)
    print(gauss)
    print(gauss2)

    minmax_expsf = numpy.min(expsf.data), numpy.max(expsf.data)
    minmax_empsf = numpy.min(empsf.data), numpy.max(empsf.data)
    minmax_obsvol = numpy.min(obsvol.data), numpy.max(obsvol.data)
    minmax_gauss = numpy.min(gauss.data), numpy.max(gauss.data)
    print(f"minmax_expsf: {minmax_expsf}")
    print(f"minmax_empsf: {minmax_empsf}")
    print(f"minmax_obsvol: {minmax_obsvol}")
    print(f"minmax_gauss: {minmax_gauss}")

    if savebin:
        # save zr slices to BIN files
        empsf.data.tofile('empsf.bin')
        expsf.data.tofile('expsf.bin')
        gauss.data.tofile('gauss.bin')
        obsvol.data.tofile('obsvol.bin')

    if savetif:
        # save zr slices to TIFF files
        from tifffile import imwrite

        imwrite('empsf.tif', empsf.data)
        imwrite('expsf.tif', expsf.data)
        imwrite('gauss.tif', gauss.data)
        imwrite('obsvol.tif', obsvol.data)

    if savevol:
        # save xyz volumes to files. Requires 32 GB for 512x512x512
        from tifffile import imwrite

        imwrite('empsf_vol.tif', empsf.volume())
        imwrite('expsf_vol.tif', expsf.volume())
        imwrite('gauss_vol.tif', gauss.volume())
        imwrite('obsvol_vol.tif', obsvol.volume())

    if not plot:
        return

    def _sym_norm(arr: numpy.ndarray) -> numpy.ndarray:
        """Mirror data to full field and normalize to [0, 1] for plotting."""
        mirrored = psf.mirror_symmetry(arr)
        vmax = numpy.max(mirrored)
        return mirrored / vmax if vmax > 0 else mirrored

    # Log-plot xy, and rz slices using Plotly
    fig_images = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[
            'expsf',
            'empsf',
            'obsvol',
            'gauss',
            'expsf slice',
            'empsf slice',
            'obsvol slice',
            'gauss slice',
        ],
        horizontal_spacing=0.03,
        vertical_spacing=0.1,
    )

    fig_images.add_trace(
        go.Heatmap(z=_sym_norm(expsf.data), coloraxis='coloraxis'), row=1, col=1
    )
    fig_images.add_trace(
        go.Heatmap(z=_sym_norm(empsf.data), coloraxis='coloraxis'), row=1, col=2
    )
    fig_images.add_trace(
        go.Heatmap(z=_sym_norm(obsvol.data), coloraxis='coloraxis'), row=1, col=3
    )
    fig_images.add_trace(
        go.Heatmap(z=_sym_norm(gauss.data), coloraxis='coloraxis'), row=1, col=4
    )

    i = 0
    fig_images.add_trace(
        go.Heatmap(z=_sym_norm(expsf.slice(i)), coloraxis='coloraxis'),
        row=2,
        col=1,
    )
    fig_images.add_trace(
        go.Heatmap(z=_sym_norm(empsf.slice(i)), coloraxis='coloraxis'),
        row=2,
        col=2,
    )
    fig_images.add_trace(
        go.Heatmap(z=_sym_norm(obsvol.slice(i)), coloraxis='coloraxis'),
        row=2,
        col=3,
    )
    fig_images.add_trace(
        go.Heatmap(z=_sym_norm(gauss.slice(i)), coloraxis='coloraxis'),
        row=2,
        col=4,
    )

    fig_images.update_layout(
        title='xy and rz slices (linear scale)',
        height=700,
        coloraxis={'colorscale': cmap, 'colorbar': {'title': 'intensity'}},
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # Plot cross sections
    z = numpy.arange(0, gauss.dims.ou[0], gauss.dims.ou[0] / gauss.dims.px[0])
    r = numpy.arange(0, gauss.dims.ou[1], gauss.dims.ou[1] / gauss.dims.px[1])
    zr_max = 20.0

    fig_lines = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        subplot_titles=['PSF cross sections', 'Residuals of gaussian approximation'],
        vertical_spacing=0.12,
    )

    fig_lines.add_trace(
        go.Scatter(x=r, y=expsf[0], mode='lines', name=expsf.name + ' (r)'),
        row=1,
        col=1,
    )
    fig_lines.add_trace(
        go.Scatter(x=r, y=gauss2[0], mode='lines', name=''),
        row=1,
        col=1,
    )
    fig_lines.add_trace(
        go.Scatter(x=r, y=obsvol[0], mode='lines', name=obsvol.name + ' (r)'),
        row=1,
        col=1,
    )
    fig_lines.add_trace(
        go.Scatter(x=r, y=gauss[0], mode='lines', name=''),
        row=1,
        col=1,
    )
    fig_lines.add_trace(
        go.Scatter(x=z, y=expsf[:, 0], mode='lines', name=expsf.name + ' (z)'),
        row=1,
        col=1,
    )
    fig_lines.add_trace(
        go.Scatter(x=z, y=gauss2[:, 0], mode='lines', name=''),
        row=1,
        col=1,
    )
    fig_lines.add_trace(
        go.Scatter(x=z, y=obsvol[:, 0], mode='lines', name=obsvol.name + ' (z)'),
        row=1,
        col=1,
    )
    fig_lines.add_trace(
        go.Scatter(x=z, y=gauss[:, 0], mode='lines', name=''),
        row=1,
        col=1,
    )

    fig_lines.add_trace(
        go.Scatter(
            x=r,
            y=expsf[0] - gauss2[0],
            mode='lines',
            name=expsf.name + ' (r) residual',
        ),
        row=2,
        col=1,
    )
    fig_lines.add_trace(
        go.Scatter(
            x=r,
            y=obsvol[0] - gauss[0],
            mode='lines',
            name=obsvol.name + ' (r) residual',
        ),
        row=2,
        col=1,
    )
    fig_lines.add_trace(
        go.Scatter(
            x=z,
            y=expsf[:, 0] - gauss2[:, 0],
            mode='lines',
            name=expsf.name + ' (z) residual',
        ),
        row=2,
        col=1,
    )
    fig_lines.add_trace(
        go.Scatter(
            x=z,
            y=obsvol[:, 0] - gauss[:, 0],
            mode='lines',
            name=obsvol.name + ' (z) residual',
        ),
        row=2,
        col=1,
    )

    fig_lines.update_xaxes(range=[0, zr_max], row=1, col=1)
    fig_lines.update_yaxes(range=[0, 1], row=1, col=1)
    fig_lines.update_xaxes(range=[0, zr_max], row=2, col=1)
    fig_lines.update_yaxes(range=[-0.25, 0.25], row=2, col=1)
    fig_lines.update_layout(
        height=700,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    fig_images.show()
    fig_lines.show()


if __name__ == '__main__':
    psf_example()
