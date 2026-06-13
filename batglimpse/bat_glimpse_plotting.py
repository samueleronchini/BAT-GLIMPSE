import logging
import os
from pathlib import Path

import astropy.units as u
import batanalysis as ba
from . import bat_glimpse_utils as utils
import healpy as hp
import ligo.skymap.io
import ligo.skymap.plot
import ligo.skymap.postprocess
import matplotlib.pyplot as plt
import numpy as np
import swiftbat
from astropy.modeling.models import Gaussian1D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


FERMI_PC_SAMPLE_COUNT = 1024
FERMI_PC_LC_BIN_SIZE = 0.256
FERMI_PC_LC_WINDOW = (-5.0, 15.0)


def map_ext(fit_file, workdir, ax_globe, add_legend=True):
    skymap, _ = ligo.skymap.io.fits.read_sky_map(os.path.join(workdir, fit_file), nest=True, distances=False)
    white_to_blue = LinearSegmentedColormap.from_list("whiteblue", ["white", "orange"])
    ax_globe.imshow_hpx((skymap.copy(), "ICRS"), cmap=white_to_blue, alpha=1.0, nested=True, zorder=0)
    cls = 100 * ligo.skymap.postprocess.util.find_greedy_credible_levels(skymap)
    ax_globe.contour_hpx((cls, "ICRS"), nested=True, colors="black", levels=(50, 90), zorder=0, linestyles=["dashed", "solid"], linewidths=0.7)
    if add_legend:
        loc_line = [
            Line2D([0], [0], color="red", linestyle="dotted", linewidth=2, label="External localization", alpha=0.0),
            Line2D([0], [0], color="black", linestyle="solid", linewidth=2, label="90% C.L."),
            Line2D([0], [0], color="black", linestyle="dashed", linewidth=2, label="50% C.L."),
        ]
        red_legend = ax_globe.legend(handles=loc_line, loc="lower right", frameon=True, bbox_to_anchor=(1.05, -0.18), borderaxespad=0.5)
        ax_globe.add_artist(red_legend)


def _find_ext_map_file(workdir):
    candidates = [
        fname
        for fname in os.listdir(workdir)
        if ("ext_loc" in fname or "glg_healpix" in fname)
        and fname.endswith((".fit", ".fits", ".fit.gz", ".fits.gz"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda fname: os.path.getmtime(os.path.join(workdir, fname)))


def _load_probability_map(workdir, fit_file):
    skymap, _ = ligo.skymap.io.fits.read_sky_map(os.path.join(workdir, fit_file), nest=True, distances=False)
    probability = np.asarray(skymap, dtype=float)
    probability = np.clip(probability, 0.0, None)
    total = np.nansum(probability)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Fermi probability map has non-positive total probability")
    return probability / total


def _ext_zoom_region(probability):
    nside = hp.npix2nside(probability.size)
    peak_pix = int(np.nanargmax(probability))
    peak_ra, peak_dec = hp.pix2ang(nside, peak_pix, nest=True, lonlat=True)
    cls = 100 * ligo.skymap.postprocess.util.find_greedy_credible_levels(probability)
    region_mask = np.isfinite(cls) & (cls <= 90.0)
    if not np.any(region_mask):
        return float(peak_ra), float(peak_dec), 20.0

    region_pix = np.where(region_mask)[0]
    region_ra, region_dec = hp.pix2ang(nside, region_pix, nest=True, lonlat=True)
    lon1 = np.deg2rad(float(peak_ra))
    lat1 = np.deg2rad(float(peak_dec))
    lon2 = np.deg2rad(region_ra)
    lat2 = np.deg2rad(region_dec)
    cos_sep = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    separations = np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
    radius = float(np.clip(np.nanpercentile(separations, 95) * 1.1, 5.0, 60.0))
    return float(peak_ra), float(peak_dec), radius


def _sample_ext_positions(probability, sample_count, seed):
    nside = hp.npix2nside(probability.size)
    rng = np.random.default_rng(seed)
    pixels = rng.choice(probability.size, size=sample_count, replace=True, p=probability)
    sample_ra, sample_dec = hp.pix2ang(nside, pixels, nest=True, lonlat=True)
    return np.asarray(sample_ra, dtype=float), np.asarray(sample_dec, dtype=float)


def _attitude_arrays(event):
    att_time = np.asarray(getattr(event.attitude.time, "value", event.attitude.time), dtype=float)
    att_ra = np.asarray(getattr(event.attitude.ra, "value", event.attitude.ra), dtype=float)
    att_dec = np.asarray(getattr(event.attitude.dec, "value", event.attitude.dec), dtype=float)
    att_roll = np.asarray(getattr(event.attitude.roll, "value", event.attitude.roll), dtype=float)
    settled_10 = np.asarray(getattr(event.attitude, "is_10arcmin_settled", np.ones(att_time.shape, dtype=bool)), dtype=bool)
    settled = np.asarray(getattr(event.attitude, "is_settled", np.ones(att_time.shape, dtype=bool)), dtype=bool)
    return att_time, att_ra, att_dec, att_roll, settled_10 & settled


def _nearest_attitude_index(att_time, target_time):
    valid_idx = np.where(np.isfinite(att_time))[0]
    if valid_idx.size == 0:
        raise ValueError("No finite attitude times available")
    local_idx = np.argmin(np.abs(att_time[valid_idx] - target_time))
    return int(valid_idx[local_idx])


def _build_sample_sources(sample_ra, sample_dec):
    return [
        swiftbat.source(ra=float(ra_val), dec=float(dec_val), name=f"fermi_pc_{idx}")
        for idx, (ra_val, dec_val) in enumerate(zip(sample_ra, sample_dec))
    ]


def _partial_coding_distribution(sample_sources, att_ra, att_dec, att_roll):
    pc_values = np.empty(len(sample_sources), dtype=float)
    for idx, source in enumerate(sample_sources):
        pc_values[idx] = float(source.exposure(ra=att_ra, dec=att_dec, roll=att_roll)[0]) / 5200.0
    return pc_values


def _distribution_summary(pc_values):
    finite_values = np.asarray(pc_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan, np.nan, np.nan
    p16, median, p84 = np.percentile(finite_values, [16, 50, 84])
    return float(p16), float(median), float(p84)


def _plot_weighted_partial_coding_lightcurve(workdir, t0, sample_sources, att_time, att_ra, att_dec, att_roll):
    event_file = utils.resolve_event_file(workdir)
    if event_file is None:
        logging.warning("No event file found for weighted partial-coding light curve in %s", workdir)
        return None

    event_times, gti_start, gti_stop = utils.load_event_times(event_file)
    if event_times is None or gti_start is None or gti_stop is None:
        logging.warning("Could not read EVENTS/GTI from %s for weighted partial-coding light curve", event_file)
        return None

    bin_centers, counts = utils.bin_light_curve(event_times, FERMI_PC_LC_BIN_SIZE, gti_start, gti_stop, t0)
    time_rel = np.asarray(bin_centers - t0, dtype=float)
    time_mask = (time_rel >= FERMI_PC_LC_WINDOW[0]) & (time_rel <= FERMI_PC_LC_WINDOW[1])
    if not np.any(time_mask):
        logging.warning("No 1 s light-curve bins available in [%s, %s] s for weighted partial-coding light curve", FERMI_PC_LC_WINDOW[0], FERMI_PC_LC_WINDOW[1])
        return None

    time_window = time_rel[time_mask]
    counts_window = np.asarray(counts[time_mask], dtype=float)
    counts_err = np.sqrt(np.clip(counts_window, 0.0, None))

    pc16 = []
    pc50 = []
    pc84 = []
    for rel_time in time_window:
        idx = _nearest_attitude_index(att_time, t0 + rel_time)
        p16, median, p84 = _distribution_summary(
            _partial_coding_distribution(
                sample_sources,
                float(att_ra[idx]),
                float(att_dec[idx]),
                float(att_roll[idx]),
            )
        )
        pc16.append(p16)
        pc50.append(median)
        pc84.append(p84)

    pc16 = np.asarray(pc16, dtype=float)
    pc50 = np.asarray(pc50, dtype=float)
    pc84 = np.asarray(pc84, dtype=float)
    weighted16 = counts_window * pc16
    weighted50 = counts_window * pc50
    weighted84 = counts_window * pc84

    plt.close("all")
    fig, (ax_counts, ax_weighted) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)

    ax_counts.step(time_window, counts_window, where="mid", color="gray", alpha=0.7, linewidth=1.6, label="Counts / 256 ms bin")
    ax_counts.errorbar(time_window, counts_window, yerr=counts_err, fmt="none", ecolor="gray", elinewidth=0.9, capsize=1.8, alpha=0.7)
    ax_counts.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax_counts.set_xlim(FERMI_PC_LC_WINDOW)
    counts_ymin = np.nanmin(counts_window) if counts_window.size else np.nan
    counts_ymax = np.nanmax(counts_window) if counts_window.size else np.nan
    if np.isfinite(counts_ymin) and np.isfinite(counts_ymax):
        if counts_ymax > counts_ymin:
            pad = 0.08 * (counts_ymax - counts_ymin)
            ax_counts.set_ylim(counts_ymin - pad, counts_ymax + pad)
        else:
            pad = max(1.0, 0.08 * max(abs(counts_ymax), 1.0))
            ax_counts.set_ylim(counts_ymin - pad, counts_ymax + pad)
    ax_counts.set_ylabel("Counts / bin", fontsize=14)
    ax_counts.set_title("Counts Light Curve", fontsize=15)
    ax_counts.legend(loc="upper right", fontsize=12)
    ax_counts.tick_params(axis="both", labelsize=12)

    ax_weighted.fill_between(time_window, weighted16, weighted84, step="mid", color="black", alpha=0.12, label="Counts × 16-84 percentile PC")
    ax_weighted.step(time_window, weighted50, where="mid", color="black", linewidth=2.0, label="Counts × median PC")
    ax_weighted.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax_weighted.set_xlim(FERMI_PC_LC_WINDOW)
    weighted_ymax = np.nanmax(weighted84) if weighted84.size else np.nan
    if np.isfinite(weighted_ymax) and weighted_ymax > 0:
        ax_weighted.set_ylim(0, weighted_ymax * 1.08)
    ax_weighted.set_xlabel("Time [s] (t - $t_0$)", fontsize=14)
    ax_weighted.set_ylabel("Counts × PC", fontsize=14)
    ax_weighted.set_title("Partial-Coding Weighted Light Curve", fontsize=15)
    ax_weighted.legend(loc="upper right", fontsize=12)
    ax_weighted.tick_params(axis="both", labelsize=12)

    plt.tight_layout()

    save_path = os.path.join(workdir, "lc_ext_weighted_pc.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info("Saved weighted partial-coding light curve to %s", save_path)
    return save_path


def _bat_fov_contour(event, t0, rel_time, energybins, healpix_nside=256):
    skyview = event.create_skyview(
        timebins=[rel_time - 0.5, rel_time + 0.5] * u.s,
        energybins=energybins,
        is_relative=True,
        T0=t0,
        input_dict=dict(aperture="CALDB:DETECTION", pcodethresh=0.01),
    )
    pcode_img_bat = ba.bat_skyimage.BatSkyImage.from_file(skyview.pcodeimg_file)
    hist = pcode_img_bat.healpix_projection(coordsys="icrs", nside=healpix_nside)
    return ba.bat_skyimage.BatSkyImage(
        image_data=hist.slice[:, :, :],
        image_type=skyview.snr_img.image_type,
        is_mosaic_intermediate=skyview.snr_img.is_mosaic_intermediate,
    ).project("HPX").contents


def _plot_bat_fov_contours(
    ax_globe,
    ax_zoom_rect,
    contour_data,
    levels,
    colors,
    linestyles,
    linewidths=2,
    zorder=2,
    add_labels=False,
    label_fmt="%1.2f",
    label_fontsize=8,
):
    level_style = sorted(
        zip(levels, colors, linestyles),
        key=lambda item: item[0],
    )
    sorted_levels = [item[0] for item in level_style]
    sorted_colors = [item[1] for item in level_style]
    sorted_linestyles = [item[2] for item in level_style]

    contour_globe = ax_globe.contour_hpx(
        (contour_data, "ICRS"),
        nested=False,
        colors=sorted_colors,
        levels=sorted_levels,
        linewidths=linewidths,
        linestyles=sorted_linestyles,
        zorder=zorder,
    )
    ax_zoom_rect.contour_hpx(
        (contour_data, "ICRS"),
        nested=False,
        colors=sorted_colors,
        levels=sorted_levels,
        linewidths=linewidths,
        linestyles=sorted_linestyles,
        zorder=zorder,
    )
    if add_labels:
        ax_globe.clabel(contour_globe, inline=True, fmt=label_fmt, fontsize=label_fontsize)


def plot_ext_bat_diagnostic(event, t0, workdir, energybins=[15, 350] * u.keV, sample_count=FERMI_PC_SAMPLE_COUNT):
    try:
        fit_file = _find_ext_map_file(workdir)
        if fit_file is None:
            logging.warning("No external localization map found in %s; skipping pre-run BAT/External diagnostic.", workdir)
            return None

        probability = _load_probability_map(workdir, fit_file)
        center_ra, center_dec, zoom_radius = _ext_zoom_region(probability)
        sample_ra, sample_dec = _sample_ext_positions(probability, sample_count, seed=int(abs(float(t0))) % (2**32))
        sample_sources = _build_sample_sources(sample_ra, sample_dec)

        att_time, att_ra, att_dec, att_roll, settled_mask = _attitude_arrays(event)
        t0_idx = _nearest_attitude_index(att_time, t0)
        is_slew_at_t0 = not bool(settled_mask[t0_idx])
        eval_offsets = [-5.0, 0.0, 5.0, 10.0, 15.0] if is_slew_at_t0 else [0.0]

        pc_distributions = {}
        actual_offsets = {}
        for offset in eval_offsets:
            idx = _nearest_attitude_index(att_time, t0 + offset)
            actual_offsets[offset] = float(att_time[idx] - t0)
            pc_distributions[offset] = _partial_coding_distribution(
                sample_sources,
                float(att_ra[idx]),
                float(att_dec[idx]),
                float(att_roll[idx]),
            )

        pc_t0 = np.asarray(pc_distributions[0.0], dtype=float)
        pc_t0_finite = pc_t0[np.isfinite(pc_t0)]
        if pc_t0_finite.size == 0:
            raise ValueError("Partial-coding distribution at t0 contains no finite values")

        fov_t0 = _bat_fov_contour(event, t0, 0.0, energybins)
        fov_tminus15 = _bat_fov_contour(event, t0, -15.0, energybins) if is_slew_at_t0 else None
        fov_tplus15 = _bat_fov_contour(event, t0, 15.0, energybins) if is_slew_at_t0 else None

        plt.close("all")
        fig = plt.figure()
        ax_zoom_rect = plt.axes(
            [-1.18, 0.02, 0.84, 0.84],
            projection="astro degrees zoom",
            center=f"{center_ra}d {center_dec}d",
            radius=f"{zoom_radius} deg",
        )
        ax_zoom_rect.coords[0].set_major_formatter("d.d")
        ax_zoom_rect.coords[1].set_major_formatter("d.d")
        ax_zoom_rect.grid()

        ax_globe = plt.axes([-0.27, -0.01, 1.16, 1.16], projection="astro degrees mollweide")
        ax_globe.grid()
        ax_globe.mark_inset_axes(ax_zoom_rect)
        ax_globe.connect_inset_axes(ax_zoom_rect, "upper right")
        ax_globe.connect_inset_axes(ax_zoom_rect, "lower right")

        if is_slew_at_t0:
            ax_hist = fig.add_axes([1.08, 0.60, 0.5, 0.34])
            ax_stats = fig.add_axes([1.08, 0.14, 0.5, 0.34])
        else:
            ax_hist = fig.add_axes([1.08, 0.22, 0.5, 0.72])
            ax_stats = None

        map_ext(fit_file, workdir, ax_globe)
        map_ext(fit_file, workdir, ax_zoom_rect, add_legend=False)

        pc_vmin = float(np.nanmin(pc_t0_finite))
        pc_vmax = float(np.nanmax(pc_t0_finite))
        if pc_vmax <= pc_vmin:
            pc_vmax = pc_vmin + 1e-3

        scatter_zoom = ax_zoom_rect.scatter(
            sample_ra,
            sample_dec,
            c=pc_t0,
            s=12,
            cmap="viridis",
            vmin=pc_vmin,
            vmax=pc_vmax,
            alpha=0.35,
            linewidths=0,
            transform=ax_zoom_rect.get_transform("world"),
            zorder=2,
        )

        if is_slew_at_t0:
            for contour_data, color, linestyle in [
                (fov_tminus15, "orange", "solid"),
                (fov_t0, "lightblue", "solid"),
                (fov_tplus15, "green", "solid"),
            ]:
                _plot_bat_fov_contours(
                    ax_globe,
                    ax_zoom_rect,
                    contour_data,
                    levels=[0.01],
                    colors=[color],
                    linestyles=[linestyle],
                )
        else:
            _plot_bat_fov_contours(
                ax_globe,
                ax_zoom_rect,
                fov_t0,
                levels=[0.8, 0.3, 0.01],
                colors=["red", "red", "red"],
                linestyles=["solid", "solid", "solid"],
                linewidths=1,
                add_labels=True,
            )

        cbar = plt.colorbar(scatter_zoom, ax=ax_zoom_rect, shrink=0.5, orientation="horizontal", aspect=30, pad=0.15)
        cbar.mappable.set_clim(vmin=pc_vmin, vmax=pc_vmax)
        cbar.set_label("Partial coding @ $t_0$")

        hist_values = pc_t0_finite
        ax_hist.hist(hist_values, bins=50, alpha=0.8, color="blue", label="PC distribution @ $t_0$", histtype="step")
        p16_t0, median_t0, p84_t0 = _distribution_summary(hist_values)
        ax_hist.axvspan(p16_t0, p84_t0, color="red", alpha=0.12, label="16-84 percentile")
        ax_hist.axvline(median_t0, color="red", linestyle="--", label=f"Median: {median_t0:.3f}")
        ax_hist.set_xlabel("Partial coding")
        ax_hist.set_ylabel("Samples")
        ax_hist.set_xlim(min(0.0, float(np.nanmin(hist_values)) - 0.02), max(1.05, float(np.nanmax(hist_values)) + 0.05))
        ax_hist.legend(loc="upper right")

        if is_slew_at_t0:
            fov_line = [
                Line2D([0], [0], color="orange", linestyle="solid", linewidth=2, label="BAT 1% PC @ $t_0$ - 15 s"),
                Line2D([0], [0], color="lightblue", linestyle="solid", linewidth=2, label="BAT 1% PC @ $t_0$"),
                Line2D([0], [0], color="green", linestyle="solid", linewidth=2, label="BAT 1% PC @ $t_0$ + 15 s"),
            ]
        else:
            fov_line = [
                Line2D([0], [0], color="red", linestyle="solid", linewidth=2, label="BAT partial coding contours @ $t_0$"),
            ]
        fov_legend = ax_globe.legend(handles=fov_line, loc="lower left", frameon=True, bbox_to_anchor=(-0.05, -0.18), borderaxespad=0.5)
        ax_globe.add_artist(fov_legend)

        if ax_stats is not None:
            stat_offsets = []
            medians = []
            lower_band = []
            upper_band = []
            for offset in eval_offsets:
                p16, median, p84 = _distribution_summary(pc_distributions[offset])
                if not np.isfinite(median):
                    continue
                stat_offsets.append(offset)
                medians.append(median)
                lower_band.append(p16)
                upper_band.append(p84)

            stat_offsets = np.asarray(stat_offsets, dtype=float)
            medians = np.asarray(medians, dtype=float)
            lower_band = np.asarray(lower_band, dtype=float)
            upper_band = np.asarray(upper_band, dtype=float)

            if stat_offsets.size > 0:
                if stat_offsets.size > 1:
                    interp_offsets = np.linspace(stat_offsets.min(), stat_offsets.max(), 256)
                    interp_lower = np.interp(interp_offsets, stat_offsets, lower_band)
                    interp_median = np.interp(interp_offsets, stat_offsets, medians)
                    interp_upper = np.interp(interp_offsets, stat_offsets, upper_band)
                else:
                    interp_offsets = stat_offsets
                    interp_lower = lower_band
                    interp_median = medians
                    interp_upper = upper_band

                ax_stats.fill_between(interp_offsets, interp_lower, interp_upper, color="black", alpha=0.15, label="16-84 percentile")
                ax_stats.plot(interp_offsets, interp_median, color="black", linewidth=1.8, label="Median")
                ax_stats.scatter(stat_offsets, medians, color="black", s=22, zorder=3)
                ax_stats.axvline(0.0, color="red", linestyle="--", linewidth=1)
                ax_stats.set_xlim(min(eval_offsets) - 1, max(eval_offsets) + 1)
                ymin = float(np.nanmin(lower_band))
                ymax = float(np.nanmax(upper_band))
                if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                    pad = 0.05 * (ymax - ymin)
                    ax_stats.set_ylim(ymin - pad, ymax + pad)
                ax_stats.set_xlabel("Time [s] (t - $t_0$)")
                ax_stats.set_ylabel("Partial coding")
                ax_stats.legend(loc="best")

        save_path = os.path.join(workdir, "map_ext_bat_pc.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        weighted_lc_path = _plot_weighted_partial_coding_lightcurve(
            workdir,
            t0,
            sample_sources,
            att_time,
            att_ra,
            att_dec,
            att_roll,
        )

        logging.info(
            "Saved Fermi/BAT diagnostic to %s and weighted light curve to %s using map %s (slew_at_t0=%s, requested_offsets=%s, actual_offsets=%s)",
            save_path,
            weighted_lc_path,
            fit_file,
            is_slew_at_t0,
            eval_offsets,
            actual_offsets,
        )
        return save_path
    except Exception:
        logging.error("Error in plot_ext_bat_diagnostic", exc_info=True)
        return None


def plot_snr(skyview, dt, duration, e1, e2, workdir, flag):
    val, _, _ = plt.hist(skyview.snr_img.contents.flatten(), bins=100, alpha=0.5, color="blue", label="SNR histogram")
    plt.xlabel("SNR")
    plt.ylabel("Counts")
    h = skyview.snr_img.contents.flatten()
    max_snr_value = np.max(h)
    plt.axvline(max_snr_value, color="red", linestyle="--", label=f"Max SNR: {max_snr_value:.2f}")
    g = Gaussian1D(amplitude=val.max(), stddev=1)
    x = np.arange(-10, 10, 0.01)
    plt.plot(x, g(x), "k-")
    plt.yscale("log")
    plt.ylim([0.8, 2 * val.max()])
    base_filename = f"hist_dt_{round(dt, 2)}_dur_{round(duration, 2)}_e1_{round(e1, 2)}_e2_{round(e2, 2)}_{flag}.png"
    save_path = os.path.join(workdir, base_filename)
    counter = 1
    while os.path.exists(save_path):
        base_filename = f"hist_dt_{round(dt, 2)}_dur_{round(duration, 2)}_e1_{round(e1, 2)}_e2_{round(e2, 2)}_{flag}_{counter}.png"
        save_path = os.path.join(workdir, base_filename)
        counter += 1
    plt.savefig(save_path)
    plt.close()


def plot_lc(event, ra, dec, t0, deltat, workdir, pipe):
    event.apply_mask_weighting(ra=ra * u.deg, dec=dec * u.deg)
    lc = event.create_lightcurve(
        lc_file=Path(f"plot_lc_{pipe}.lc"),
        energybins=[15, 50, 100, 350] * u.keV,
        recalc=True,
    )
    color = ["red", "blue", "green", "black"]
    label = ["15-50 keV", "50-100 keV", "100-350 keV", "15-350 keV"]
    bin_sizes = [0.064, 0.256, 1.024, 4.096]
    fig, axes = plt.subplots(len(bin_sizes), 1, figsize=(10, 5 * len(bin_sizes)), sharex=True)
    for i, bin_size in enumerate(bin_sizes):
        ax = axes[i]
        lc.set_timebins(timebinalg="uniform", timedelta=np.timedelta64(int(bin_size * 1000), "ms"), tmin=-5 * deltat * u.s, tmax=10 * deltat * u.s, T0=t0, is_relative=True)
        lc.set_energybins(energybins=[15, 50, 100, 350] * u.keV)
        for n in range(4):
            time = lc.data["TIME"].value
            counts = lc.data["RATE"][:, n].value
            errors = lc.data["ERROR"][:, n].value
            if n == 3:
                ax.errorbar(time - t0, counts, yerr=errors, color=color[n], ls="", zorder=n, alpha=1.0)
                ax.step(time - t0, counts, where="mid", color=color[n], label=label[n], zorder=n, alpha=1.0)
            else:
                ax.step(time - t0, counts, where="mid", color=color[n], label=label[n], zorder=n, alpha=0.4)
        ax.set_title(f"Light Curve (bin size: {bin_size:.3f} s)")
        ax.axhline(y=0.0, color="r", linestyle="--", linewidth=1)
        ax.set_ylabel("Counts/bin")
        ax.legend()
    axes[-1].set_xlabel("Time [s] (t - t0)")
    plt.tight_layout()
    plt.savefig(os.path.join(workdir, f"lc_{pipe}.png"), dpi=300)
    plt.close()


def pc_time(ra, dec, event, t0, timebins, workdir):
    try:
        plt.close("all")
        plt.figure()
        object_batsource = swiftbat.source(ra=ra, dec=dec, name="pc_time")
        time = event.attitude.time.value - t0 * np.ones(len(event.attitude.time))
        exposures = np.array([object_batsource.exposure(ra=ra_i, dec=dec_i, roll=roll)[0] for ra_i, dec_i, roll in zip(event.attitude.ra, event.attitude.dec, event.attitude.roll)])
        plt.plot(time, exposures / 5200)
        plt.xlim(-20, 50)
        time_window_mask = (time >= -20) & (time <= 50)
        plt.ylim(min(exposures[time_window_mask] / 5200) * 0.95, max(exposures[time_window_mask] / 5200) * 1.05)
        plt.axvspan(timebins[0].value, timebins[1].value, color="gray", alpha=0.3, label="Period analysed with mosaic")
        plt.legend()
        plt.xlabel("t - T0 (s)")
        plt.ylabel("Partial coding @ source position")
        plt.savefig(os.path.join(workdir, "pc_time.png"), dpi=300)
    except Exception:
        logging.error("Error in pc_time")


def map_mosaic(event, ra_s, dec_s, time_bins, energybins, t0, workdir):
    slew_skyviews = ba.parallel.create_event_skyview(event, timebins=time_bins, energybins=energybins, is_relative=True, parse_images=False, T0=t0, nprocs=10)
    mosaic_skyview = ba.parallel.mosaic_skyview(slew_skyviews, healpix_nside=1024, healpix_coordsys="icrs", nprocs=10)
    plt.hist(mosaic_skyview.snr_img.contents.flatten(), bins=100, alpha=0.5, color="blue", label="SNR histogram")
    t = mosaic_skyview.snr_img
    hist = t.healpix_projection(coordsys="icrs", nside=1024)
    plot_quantity = ba.bat_skyimage.BatSkyImage(image_data=hist.slice[:, :, :], image_type=t.image_type, is_mosaic_intermediate=t.is_mosaic_intermediate).project("HPX").contents
    max_snr_value = np.nanmax(mosaic_skyview.snr_img.contents.flatten())
    min_snr_value = np.nanmin(mosaic_skyview.snr_img.contents.flatten())
    fig = plt.figure()
    ax_zoom_rect = plt.axes([-1.2, -0.0, 0.9, 0.9], projection="astro degrees zoom", center=f"{ra_s}d {dec_s}d", radius="2 deg")
    ax_zoom_rect.coords[0].set_major_formatter("d.d")
    ax_zoom_rect.coords[1].set_major_formatter("d.d")
    ax_zoom_rect.grid()
    ax_hist = fig.add_axes([1.1, 0.2, 0.5, 0.8])
    val, _, _ = plt.hist(mosaic_skyview.snr_img.contents.flatten(), bins=100, alpha=0.8, color="blue", label="SNR histogram", histtype="step")
    ax_hist.axvline(max_snr_value, color="red", linestyle="--", label=f"Max SNR: {max_snr_value:.2f}")
    g = Gaussian1D(amplitude=val.max(), stddev=1)
    x = np.arange(-30, 30, 0.01)
    ax_hist.plot(x, g(x), "k-")
    ax_hist.set_xlabel("")
    ax_hist.set_ylabel("SNR")
    ax_hist.set_yscale("log")
    ax_hist.yaxis.tick_right()
    ax_hist.yaxis.set_label_position("right")
    ax_hist.set_ylim([0.1, 2 * val.max()])
    ax_hist.set_xlim([min_snr_value - 1, max_snr_value + 1])
    ax_hist.legend(loc="upper right")
    vmin = np.nanmin(plot_quantity)
    vmax = np.nanmax(plot_quantity)
    cmap_custom = plt.cm.magma
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=norm)
    cbar = plt.colorbar(sm, ax=ax_zoom_rect, shrink=0.5, orientation="horizontal", aspect=30, pad=0.15)
    cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
    cbar.set_label("SNR")
    ax_globe = plt.axes([-0.3, -0.05, 1.25, 1.25], projection="astro degrees mollweide")
    ax_globe.grid()
    ax_globe.mark_inset_axes(ax_zoom_rect)
    ax_globe.connect_inset_axes(ax_zoom_rect, "upper right")
    ax_globe.connect_inset_axes(ax_zoom_rect, "lower right")
    pcc_mapp_arr = []
    time_bins = [[-10, -9], [-1, +1], [10, 11]] * u.s
    for n in range(3):
        settled_skyview = event.create_skyview(timebins=time_bins[n], energybins=energybins, is_relative=True, T0=t0)
        pcode_img_bat = ba.bat_skyimage.BatSkyImage.from_file(settled_skyview.pcodeimg_file)
        hist = pcode_img_bat.healpix_projection(coordsys="icrs", nside=256)
        pcc_mapp_arr.append(ba.bat_skyimage.BatSkyImage(image_data=hist.slice[:, :, :], image_type=t.image_type, is_mosaic_intermediate=t.is_mosaic_intermediate).project("HPX").contents)
    pcc_mapp_arr = np.array(pcc_mapp_arr)
    col = ["orange", "lightblue", "green"]
    ls = ["solid", "solid", "solid"]
    for n in range(3):
        ax_globe.contour_hpx((pcc_mapp_arr[n], "ICRS"), nested=False, colors=col[n], levels=[0.1], linewidths=2, linestyles=ls[n])
    ax_zoom_rect.imshow_hpx((plot_quantity, "ICRS"), cmap="magma", alpha=1.0, zorder=1)
    fov_line = [
        Line2D([0], [0], color="orange", linestyle="solid", linewidth=2, label="FOV @ $t_0$ - 10 s"),
        Line2D([0], [0], color="lightblue", linestyle="solid", linewidth=2, label="FOV @ $t_0$"),
        Line2D([0], [0], color="green", linestyle="solid", linewidth=2, label="FOV @ $t_0$ + 10 s"),
    ]
    fov_legend = ax_globe.legend(handles=fov_line, loc="lower left", frameon=True, bbox_to_anchor=(-0.05, -0.18), borderaxespad=0.5)
    ax_globe.add_artist(fov_legend)
    fit_file = next((fname for fname in os.listdir(workdir) if "ext_loc" in fname), None)
    if fit_file:
        from .bat_glimpse_plotting import map_ext
        map_ext(fit_file, workdir, ax_globe)
    plt.savefig(os.path.join(workdir, "map_mosaic.png"), bbox_inches="tight", dpi=300)


def map_imaging(event, ra_s, dec_s, time_bins, energybins, t0, workdir):
    skyview = event.create_skyview(timebins=time_bins, energybins=energybins, is_relative=True, T0=t0, input_dict=dict(aperture="CALDB:DETECTION"))
    t = skyview.snr_img
    hist = t.healpix_projection(coordsys="icrs", nside=1024)
    plot_quantity = ba.bat_skyimage.BatSkyImage(image_data=hist.slice[:, :, :], image_type=t.image_type, is_mosaic_intermediate=t.is_mosaic_intermediate).project("HPX").contents
    fig = plt.figure()
    ax_zoom_rect = plt.axes([-1.2, 0.0, 0.9, 0.9], projection="astro degrees zoom", center=f"{ra_s}d {dec_s}d", radius="2 deg")
    ax_zoom_rect.coords[0].set_major_formatter("d.d")
    ax_zoom_rect.coords[1].set_major_formatter("d.d")
    ax_zoom_rect.grid()
    vmin = np.nanmin(plot_quantity)
    vmax = np.nanmax(plot_quantity)
    cmap_custom = plt.cm.inferno
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=norm)
    cbar = plt.colorbar(sm, ax=ax_zoom_rect, shrink=0.5, orientation="horizontal", aspect=30, pad=0.15)
    cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
    cbar.set_label("SNR")
    ax_globe = plt.axes([-0.3, -0.05, 1.25, 1.25], projection="astro degrees mollweide")
    ax_globe.grid()
    ax_globe.mark_inset_axes(ax_zoom_rect)
    ax_globe.connect_inset_axes(ax_zoom_rect, "upper right")
    ax_globe.connect_inset_axes(ax_zoom_rect, "lower right")
    ax_hist = fig.add_axes([1.1, 0.2, 0.5, 0.8])
    h = skyview.snr_img.contents.flatten()
    val, _, _ = plt.hist(h, bins=100, alpha=0.8, color="blue", label="SNR histogram", histtype="step")
    max_snr_value = np.nanmax(h)
    min_snr_value = np.nanmin(h)
    ax_hist.axvline(max_snr_value, color="red", linestyle="--", label=f"Max SNR: {max_snr_value:.2f}")
    g = Gaussian1D(amplitude=val.max(), stddev=1)
    x = np.arange(-30, 30, 0.01)
    ax_hist.plot(x, g(x), "k-")
    ax_hist.set_xlabel("")
    ax_hist.set_ylabel("SNR")
    ax_hist.set_yscale("log")
    ax_hist.yaxis.tick_right()
    ax_hist.yaxis.set_label_position("right")
    ax_hist.set_ylim([0.1, 2 * val.max()])
    ax_hist.set_xlim([min_snr_value - 1, max_snr_value + 1])
    ax_hist.legend(loc="upper right")
    fit_file = next((fname for fname in os.listdir(workdir) if "ext_loc" in fname), None)
    if fit_file:
        map_ext(fit_file, workdir, ax_globe)
    pcode_img_bat = ba.bat_skyimage.BatSkyImage.from_file(skyview.pcodeimg_file)
    hist = pcode_img_bat.healpix_projection(coordsys="icrs", nside=1024)
    plot_quantity_pc = ba.bat_skyimage.BatSkyImage(image_data=hist.slice[:, :, :], image_type=t.image_type, is_mosaic_intermediate=t.is_mosaic_intermediate).project("HPX").contents
    cs = ax_globe.contour_hpx((plot_quantity_pc, "ICRS"), nested=False, colors="red", levels=[0.01, 0.3, 0.8], linewidths=1, zorder=0)
    ax_globe.clabel(cs, inline=True, fmt="%1.2f", fontsize=8)
    ax_zoom_rect.imshow_hpx((plot_quantity, "ICRS"), cmap="magma", alpha=1.0, zorder=1)
    red_line = [Line2D([0], [0], color="red", lw=1, label="Partial coding")]
    red_legend = ax_globe.legend(handles=red_line, loc="lower left", frameon=True, bbox_to_anchor=(-0.05, -0.10), borderaxespad=0.5)
    ax_globe.add_artist(red_legend)
    plt.savefig(os.path.join(workdir, "map_imaging.png"), bbox_inches="tight", dpi=300)
