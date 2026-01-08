# psf_example.py

"""Point Spread Function example.

Demonstrate the use of the psf library for calculating point spread functions
for fluorescence microscopy.

"""

from __future__ import annotations

from typing import Any

import numpy
import psf
from matplotlib import pyplot
from matplotlib.widgets import Slider, Button, RadioButtons


def psf_example(
    cmap: str = 'copper',
    savebin: bool = False,
    savetif: bool = False,
    savevol: bool = False,
    plot: bool = True,
    **kwargs: Any,
) -> None:
    """Calculate, save, and plot various point spread functions."""
    args = {
        'shape': (52, 52),  # number of samples in z and r direction
        'dims': (5.2, 5.2),  # total size in z and r direction in micrometers
        # size in z and r direction should be 5.2 micrometers, so in xyz space, total size should be 5.2*2 ~= 10.3 micrometers
        'ex_wavelen': 488.0,  # excitation wavelength in nanometers
        'em_wavelen': 520.0,  # emission wavelength in nanometers
        'num_aperture': 0.9,
        # 'num_aperture': 1.2, # default value
        'refr_index': 1.0,
        # 'refr_index': 1.333, # default value
        'magnification': 1.0,
        'pinhole_radius': 0.05,  # in micrometers
        'pinhole_shape': 'square',
    }
    args.update(kwargs)

    def _mirrored_log(arr: numpy.ndarray) -> numpy.ndarray:
        """Apply mirror symmetry and log10 for plotting."""
        return psf.mirror_symmetry(numpy.log10(numpy.maximum(arr, 1e-15)))

    def _compute(na: float, ri: float, mode: str = 'confocal'):
        a = dict(args)
        a['num_aperture'] = na
        a['refr_index'] = ri
        if mode == 'widefield':
            # For widefield: excitation PSF is uniform (all 1), detection is emission PSF
            obs = psf.PSF(psf.ISOTROPIC | psf.WIDEFIELD, **a)  # type: ignore[arg-type]
            assert obs.expsf is not None
            assert obs.empsf is not None
            # Create uniform excitation PSF (all 1) with same shape as emission
            uniform_ex = psf.PSF.__new__(psf.PSF)
            uniform_ex.data = numpy.ones_like(obs.empsf.data)
            uniform_ex.shape = obs.empsf.shape
            uniform_ex.dims = obs.empsf.dims
            uniform_ex.name = 'Widefield Excitation'
            # Add slice method for uniform PSF - return all 1s with correct shape
            import psf._psf as _psf  # noqa: PLC0415
            def uniform_slice(key=slice(None)):
                # Get the shape of the slice from emission PSF to match dimensions
                em_slice = obs.empsf.slice(key)
                # Return all 1s with same shape
                return numpy.ones_like(em_slice)
            uniform_ex.slice = uniform_slice
            # Add volume method for uniform PSF - return all 1s with correct 3D shape
            def uniform_volume():
                em_vol = obs.empsf.volume()
                return numpy.ones_like(em_vol)
            uniform_ex.volume = uniform_volume
            # Effective PSF for widefield = excitation (1) * detection (emission) = emission
            effective = psf.PSF.__new__(psf.PSF)
            effective.data = obs.empsf.data.copy()  # effective = emission for widefield
            effective.shape = obs.empsf.shape
            effective.dims = obs.empsf.dims
            effective.name = 'Widefield Effective'
            # Add volume method for effective PSF
            effective.volume = lambda: obs.empsf.volume()
            return effective, uniform_ex, obs.empsf
        else:  # confocal
            obs = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **a)  # type: ignore[arg-type]
            assert obs.expsf is not None
            assert obs.empsf is not None
            return obs, obs.expsf, obs.empsf

    # Initial mode
    current_mode = 'confocal'
    obsvol, expsf, empsf = _compute(args['num_aperture'], args['refr_index'], current_mode)

    if savebin:
        empsf.data.tofile('empsf.bin')
        expsf.data.tofile('expsf.bin')
        obsvol.data.tofile('obsvol.bin')

    if savetif:
        from tifffile import imwrite

        imwrite('empsf.tif', empsf.data)
        imwrite('expsf.tif', expsf.data)
        imwrite('obsvol.tif', obsvol.data)

    if savevol:
        from tifffile import imwrite

        imwrite('empsf_vol.tif', empsf.volume())
        imwrite('expsf_vol.tif', expsf.volume())
        imwrite('obsvol_vol.tif', obsvol.volume())

    if not plot:
        return

    # GUI with sliders for NA and refractive index, showing PSF maps
    pyplot.rc('font', family='sans-serif', weight='normal')
    # Leave room at top for info text; 2x4 grid, leftmost column kept for controls
    fig = pyplot.figure(figsize=(12, 6), dpi=96)
    gs = fig.add_gridspec(2, 4, width_ratios=[0.6, 1, 1, 1], hspace=0.08, wspace=0.12)
    fig.subplots_adjust(top=0.82)
    axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
    ]

    im_ex = axes[3].imshow(_mirrored_log(expsf.data), cmap=cmap, vmin=-2.5, vmax=0)
    im_em = axes[4].imshow(_mirrored_log(empsf.data), cmap=cmap, vmin=-2.5, vmax=0)
    im_ob = axes[5].imshow(_mirrored_log(obsvol.data), cmap=cmap, vmin=-2.5, vmax=0)

    i = 0
    im_ex_slice = axes[0].imshow(
        _mirrored_log(expsf.slice(i)), cmap=cmap, vmin=-2.5, vmax=0
    )
    im_em_slice = axes[1].imshow(
        _mirrored_log(empsf.slice(i)), cmap=cmap, vmin=-2.5, vmax=0
    )
    im_ob_slice = axes[2].imshow(
        _mirrored_log(obsvol.slice(i)), cmap=cmap, vmin=-2.5, vmax=0
    )
    for ax in axes:
        ax.axis('off')

    # Slider axes
    # Reserve top area for info text; place sliders vertically at left
    info_ax = fig.add_axes([0.12, 0.86, 0.76, 0.05])
    info_ax.axis('off')
    slider_ax_na = fig.add_axes([0.02, 0.35, 0.03, 0.40])  # left, vertical
    slider_ax_ri = fig.add_axes([0.07, 0.35, 0.03, 0.40])  # left, vertical
    mode_ax = fig.add_axes([0.02, 0.20, 0.08, 0.10])  # mode selector
    reset_ax = fig.add_axes([0.02, 0.12, 0.08, 0.04])
    save_ax = fig.add_axes([0.02, 0.06, 0.08, 0.04])

    s_na = Slider(
        slider_ax_na,
        'NA',
        0.6,
        1.49,
        valinit=args['num_aperture'],
        valstep=0.01,
        orientation='vertical',
    )
    s_ri = Slider(
        slider_ax_ri,
        'n (IOR)',
        1.0,
        1.6,
        valinit=args['refr_index'],
        valstep=0.005,
        orientation='vertical',
    )
    mode_radio = RadioButtons(mode_ax, ('Confocal', 'Widefield'), active=0)
    reset_btn = Button(reset_ax, 'Reset')
    save_btn = Button(save_ax, 'Save PSF')

    info_text = info_ax.text(
        0, 0.5, '', va='center', ha='left', fontsize=9, color='k'
    )

    def update(_):
        nonlocal current_mode
        na = float(s_na.val)
        ri = float(s_ri.val)
        # Get mode from radio button
        mode_label = mode_radio.value_selected
        current_mode = 'widefield' if mode_label == 'Widefield' else 'confocal'
        obs, ex, em = _compute(na, ri, current_mode)
        im_ex.set_data(_mirrored_log(ex.data))
        im_em.set_data(_mirrored_log(em.data))
        im_ob.set_data(_mirrored_log(obs.data))
        im_ex_slice.set_data(_mirrored_log(ex.slice(i)))
        im_em_slice.set_data(_mirrored_log(em.slice(i)))
        im_ob_slice.set_data(_mirrored_log(obs.slice(i)))
        info_text.set_text(f'Mode: {mode_label} | NA = {na:.3f}    n = {ri:.3f}')
        if current_mode == 'widefield':
            axes[0].set_title(f'Widefield Excitation PSF (XY) | NA={na:.3f}, n={ri:.3f}')
            axes[1].set_title(f'Widefield Detection PSF (XY) | NA={na:.3f}, n={ri:.3f}')
            axes[2].set_title(f'Widefield Effective PSF (XY) | NA={na:.3f}, n={ri:.3f}')
            axes[3].set_title(f'Widefield Excitation PSF (XZ) | NA={na:.3f}, n={ri:.3f}')
            axes[4].set_title(f'Widefield Detection PSF (XZ) | NA={na:.3f}, n={ri:.3f}')
            axes[5].set_title(f'Widefield Effective PSF (XZ) | NA={na:.3f}, n={ri:.3f}')
        else:
            axes[0].set_title(f'Excitation PSF (XY) | NA={na:.3f}, n={ri:.3f}')
            axes[1].set_title(f'Emission PSF (XY) | NA={na:.3f}, n={ri:.3f}')
            axes[2].set_title(f'Confocal PSF (XY) | NA={na:.3f}, n={ri:.3f}')
            axes[3].set_title(f'Excitation PSF (XZ) | NA={na:.3f}, n={ri:.3f}')
            axes[4].set_title(f'Emission PSF (XZ) | NA={na:.3f}, n={ri:.3f}')
            axes[5].set_title(f'Confocal PSF (XZ) | NA={na:.3f}, n={ri:.3f}')
        fig.canvas.draw_idle()

    def reset(event):  # noqa: ARG001
        s_na.reset()
        s_ri.reset()

    def save_psf(event):  # noqa: ARG001
        na = float(s_na.val)
        ri = float(s_ri.val)
        from tifffile import imwrite
        import os
        
        # Create directory if it doesn't exist
        os.makedirs('psf/examples/psf_data', exist_ok=True)
        # Save confocal PSF (3D volumes)
        conf_obs, conf_ex, conf_em = _compute(na, ri, 'confocal')
        # Save widefield PSF (3D volumes)
        wf_obs, wf_ex, wf_em = _compute(na, ri, 'widefield')

        print('confocal psf shape: ', conf_ex.volume().shape)
        print('confocal psf shape: ', conf_em.volume().shape)
        print('confocal psf shape: ', conf_obs.volume().shape)
        print('widefield psf shape: ', wf_ex.volume().shape)
        print('widefield psf shape: ', wf_em.volume().shape)
        print('widefield psf shape: ', wf_obs.volume().shape)

        # 3D center crop from (103,103,103) to (52,52,52) 
        center_crop_size = 52

        center_crop_start = (103 - center_crop_size) // 2
        center_crop_end = center_crop_start + center_crop_size
        conf_ex_cropped = conf_ex.volume()[center_crop_start:center_crop_end, center_crop_start:center_crop_end, center_crop_start:center_crop_end]
        conf_em_cropped = conf_em.volume()[center_crop_start:center_crop_end, center_crop_start:center_crop_end, center_crop_start:center_crop_end]
        conf_obs_cropped = conf_obs.volume()[center_crop_start:center_crop_end, center_crop_start:center_crop_end, center_crop_start:center_crop_end]
        wf_ex_cropped = wf_ex.volume()[center_crop_start:center_crop_end, center_crop_start:center_crop_end, center_crop_start:center_crop_end]
        wf_em_cropped = wf_em.volume()[center_crop_start:center_crop_end, center_crop_start:center_crop_end, center_crop_start:center_crop_end]
        wf_obs_cropped = wf_obs.volume()[center_crop_start:center_crop_end, center_crop_start:center_crop_end, center_crop_start:center_crop_end]

        print('center cropped confocal psf shape: ', conf_ex_cropped.shape)
        print('center cropped confocal psf shape: ', conf_em_cropped.shape)
        print('center cropped confocal psf shape: ', conf_obs_cropped.shape)
        print('center cropped widefield psf shape: ', wf_ex_cropped.shape)
        print('center cropped widefield psf shape: ', wf_em_cropped.shape)
        print('center cropped widefield psf shape: ', wf_obs_cropped.shape)

        imwrite('psf/examples/psf_data/widefield_excitation.tif', wf_ex_cropped)
        imwrite('psf/examples/psf_data/widefield_detection.tif', wf_em_cropped)
        imwrite('psf/examples/psf_data/widefield_effective.tif', wf_obs_cropped)
        imwrite('psf/examples/psf_data/confocal_excitation.tif', conf_ex_cropped)
        imwrite('psf/examples/psf_data/confocal_emission.tif', conf_em_cropped)
        imwrite('psf/examples/psf_data/confocal_psf.tif', conf_obs_cropped)
        
        print(f'PSF saved with NA={na:.3f}, n={ri:.3f}')
        print('Confocal (3D): confocal_excitation.tif, confocal_emission.tif, confocal_psf.tif')
        print('Widefield (3D): widefield_excitation.tif, widefield_detection.tif, widefield_effective.tif')

    s_na.on_changed(update)
    s_ri.on_changed(update)
    mode_radio.on_clicked(update)
    reset_btn.on_clicked(reset)
    save_btn.on_clicked(save_psf)
    update(None)

    pyplot.show()


if __name__ == '__main__':
    psf_example()
