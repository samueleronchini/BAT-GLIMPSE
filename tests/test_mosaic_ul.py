from types import SimpleNamespace
from unittest.mock import Mock

import astropy.units as u
import healpy as hp
import numpy as np

from batglimpse import bat_glimpse_pipeline as pipeline
from batglimpse import bat_glimpse_plotting as plotting


def test_drm_flux_and_count_rate_use_the_response():
    response = SimpleNamespace(
        input_ebins={"E_MIN": [15, 30] * u.keV, "E_MAX": [30, 60] * u.keV},
        project=lambda *args: SimpleNamespace(contents=np.diag([100.0, 200.0])),
    )
    flux, rate, _ = pipeline._drm_energy_flux_and_count_rate(
        response, e_low_keV=15, e_high_keV=60, epeak_keV=300
    )
    energies = np.array([22.5, 45.0])
    widths = np.array([15.0, 30.0])
    spectrum = (energies / 100) ** -1 * np.exp(-energies / 300)
    np.testing.assert_allclose(rate, np.sum(spectrum * [100, 200] * widths))
    np.testing.assert_allclose(flux, np.sum(spectrum * energies * widths) * 1.602176634e-9)


def test_representative_map_interpolates_drm_samples(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "_drm_time_weights", lambda *args: np.ones(1))
    monkeypatch.setattr(
        pipeline,
        "_compute_k_for_sky_position",
        lambda **kwargs: (1.0 + kwargs["ra_deg"] / 360.0, "test"),
    )
    support = np.ones(hp.nside2npix(1), dtype=bool)
    support[0] = False
    k_map, samples = pipeline._build_representative_drm_flux_factor_map(
        event=SimpleNamespace(),
        time_bins=[-4, -2] * u.s,
        energybins=[15, 350] * u.keV,
        t0=0,
        valid_skyview_indices=[0],
        workdir=str(tmp_path),
        nside=1,
        coordsys="icrs",
        pcode_support=support,
        pcode_ref=np.ones(support.size),
        bkg_map=np.ones(support.size),
        sample_count=3,
    )
    assert len(samples) == 3
    assert np.isnan(k_map[0])
    assert np.all(np.isfinite(k_map[support]))
    np.testing.assert_allclose(k_map[samples["PIX"]], samples["K"])


def test_external_localization_contours_do_not_cover_ul_map(monkeypatch):
    monkeypatch.setattr(
        plotting.ligo.skymap.io.fits, "read_sky_map", lambda *args, **kwargs: (np.ones(12), {})
    )
    monkeypatch.setattr(
        plotting.ligo.skymap.postprocess.util, "find_greedy_credible_levels", lambda values: values
    )
    axes = Mock()
    plotting.map_ext("external.fits", ".", axes, add_legend=False, contours_only=True)
    axes.imshow_hpx.assert_not_called()
    assert axes.contour_hpx.call_args.kwargs["zorder"] == 10

    axes.reset_mock()
    plotting.map_ext("external.fits", ".", axes, add_legend=False)
    axes.imshow_hpx.assert_called_once()
    assert axes.contour_hpx.call_args.kwargs["zorder"] == 0
