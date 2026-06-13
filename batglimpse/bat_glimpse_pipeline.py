import logging
import os
import shutil
import time
import traceback
from pathlib import Path

import astropy.units as u
import batanalysis as ba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, unique, vstack
from astropy.time import Time, TimeDelta
from swifttools.swift_too import Data, GUANO, ObsQuery

from . import bat_glimpse_utils as utils
from .bat_glimpse_plotting import map_imaging, map_mosaic, pc_time, plot_ext_bat_diagnostic, plot_lc, plot_snr

import subprocess
import sys

GUANO_DATA_MAX_WAIT_SECONDS = int(os.getenv("BAT_GLIMPSE_GUANO_MAX_WAIT_SECONDS", "10800"))
GUANO_RETRY_SLEEP_SECONDS = int(os.getenv("BAT_GLIMPSE_GUANO_RETRY_SLEEP_SECONDS", "60"))
GUANO_MAX_RETRIES = int(os.getenv("BAT_GLIMPSE_GUANO_MAX_RETRIES", "4"))

def log_imaging(detected_sources):
    try:
        header = "{:<3} {:<16} {:>10} {:>10} {:>8} {:>10} {:>10} {:>6}".format("Idx", "NAME", "RA", "DEC", "SNR", "CENT_SNR", "PCODEFR", "STAT")
        rows = []
        for idx, src in enumerate(detected_sources[:10]):
            status = src["DETECT_STATUS"] if "DETECT_STATUS" in src.colnames else None
            rows.append("{:<3} {:<16} {} {} {} {} {} {:>6}".format(idx, str(src["NAME"])[:15], f"{src['SKYCOORD'].ra.deg:10.3f}", f"{src['SKYCOORD'].dec.deg:10.3f}", f"{src['SNR']:8.3f}", f"{src['CENT_SNR']:10.3f}", f"{src['PCODEFR']:10.3f}", f"{int(status):6d}" if status is not None else "  None"))
        logging.info("Detected sources (first 10):\n" + "\n".join([header] + rows))
    except Exception:
        logging.error(f"Error printing detected_sources: {traceback.format_exc()}")


def log_mosaic(mosaic_detected_sources):
    try:
        header = "{:<3} {:>10} {:>10} {:>10} {:>20}".format("Idx", "SNR", "RA", "DEC", "psffwhm_separation")
        rows = []
        if mosaic_detected_sources is not None:
            for idx, src in enumerate(mosaic_detected_sources):
                coord = src.get("SNR_skycoord", None)
                snr = src.get("SNR", None)
                psf = src.get("psffwhm_separation", None)
                rows.append("{:<3} {:>10} {} {} {}".format(idx, f"{snr:10.3f}" if snr is not None else "     None", f"{coord.ra.deg:10.3f}" if coord is not None else "   None", f"{coord.dec.deg:10.3f}" if coord is not None else "   None", f"{psf:20.3f}" if psf is not None else "        None"))
            logging.info("Mosaic detected sources (SNR, coords, psffwhm_separation):\n" + "\n".join([header] + rows))
        else:
            logging.info("No mosaic detected sources found.")
    except Exception:
        logging.error(f"Error logging detected sources table: {traceback.format_exc()}")


def imaging(t0, event, workdir, min_time, max_time, seeds, time_seed):
    ra = []
    dec = []
    snr = []
    cent_snr = []
    pc = []
    det_status = []
    t_bin = []
    e1 = []
    e2 = []
    dt_det = []
    duration_det = []
    detect_params = dict(
        pcodethresh=0.01,
        snrthresh=4.5,
        # srcdetect="yes",
        # srcfit="yes",
        # posfit="yes",
        # posfluxfit="yes",
        # posfitwindow=0.05,
        # nadjpix=1,
        # nullborder="yes",
        # keepbadsources="yes",
        # srcradius=12,
        # vectorflux="no",
    )
    for energybins in [[15, 350] * u.keV]:
        if not time_seed:
            time_bins = [min_time, max_time] * u.s
            deltat = max_time - min_time
            logging.info(f"We are analyzing the temporal bin [{round(min_time, 3)},{round(max_time, 3)}] in the energy range {energybins[0].value} - {energybins[1].value} keV")
            settled_skyview = event.create_skyview(timebins=time_bins, energybins=energybins, is_relative=True, T0=t0, input_dict=dict(aperture="CALDB:DETECTION", pcodethresh=0.01), recalc=True)
            detected_sources = settled_skyview.detect_sources(input_dict=detect_params)
            log_imaging(detected_sources)
            if len(detected_sources) > 0 and any("UNKNOWN" in str(src["NAME"]) for src in detected_sources):
                plt.close("all")
                plot_snr(settled_skyview, min_time, deltat, energybins[0].value, energybins[1].value, workdir, flag="imaging")
                for src in detected_sources:
                    if "UNKNOWN" in src["NAME"] and src["SNR"] > 5 and src["CENT_SNR"] > 5:
                        ra.append(src["SKYCOORD"].ra.deg)
                        dec.append(src["SKYCOORD"].dec.deg)
                        snr.append(src["SNR"])
                        cent_snr.append(src["CENT_SNR"])
                        pc.append(src["PCODEFR"])
                        det_status.append(src["DETECT_STATUS"] if "DETECT_STATUS" in detected_sources.colnames else None)
                        t_bin.append(deltat)
                        e1.append(energybins[0].value)
                        e2.append(energybins[1].value)
        else:
            dt = seeds[0]
            duration = seeds[1]
            for n in range(len(dt)):
                time_bins = [dt[n], dt[n] + duration[n]] * u.s
                logging.info(f"We are analyzing the temporal bin [{round(dt[n], 3)},{round(dt[n] + duration[n], 3)}] in the energy range {energybins[0].value} - {energybins[1].value} keV")
                settled_skyview = event.create_skyview(timebins=time_bins, energybins=energybins, is_relative=True, T0=t0, input_dict=dict(aperture="CALDB:DETECTION", pcodethresh=0.01), recalc=True)
                detected_sources = settled_skyview.detect_sources(input_dict=detect_params)
                log_imaging(detected_sources)
                if len(detected_sources) > 0 and any("UNKNOWN" in str(src["NAME"]) for src in detected_sources):
                    plt.close("all")
                    plot_snr(settled_skyview, time_bins[0].value, time_bins[1].value - time_bins[0].value, energybins[0].value, energybins[1].value, workdir, flag="imaging")
                    for src in detected_sources:
                        if "UNKNOWN" in src["NAME"] and src["SNR"] > 5 and src["CENT_SNR"] > 5:
                            ra.append(src["SKYCOORD"].ra.deg)
                            dec.append(src["SKYCOORD"].dec.deg)
                            snr.append(src["SNR"])
                            cent_snr.append(src["CENT_SNR"])
                            pc.append(src["PCODEFR"])
                            det_status.append(src["DETECT_STATUS"] if "DETECT_STATUS" in detected_sources.colnames else None)
                            dt_det.append(dt[n])
                            duration_det.append(duration[n])
                            e1.append(energybins[0].value)
                            e2.append(energybins[1].value)
    if len(snr) > 0:
        ra_arr = np.array(ra)
        dec_arr = np.array(dec)
        snr_arr = np.array(snr)
        cent_snr_arr = np.array(cent_snr)
        pc_arr = np.array(pc)
        sorted_indices = np.argsort(snr_arr)[::-1]
        ra = ra_arr[sorted_indices].tolist()
        dec = dec_arr[sorted_indices].tolist()
        snr = snr_arr[sorted_indices].tolist()
        pc = pc_arr[sorted_indices].tolist()
        cent_snr = cent_snr_arr[sorted_indices].tolist()
        det_status = np.array(det_status)[sorted_indices].tolist()
        e1 = np.array(e1)[sorted_indices].tolist()
        e2 = np.array(e2)[sorted_indices].tolist()
        with open(os.path.join(workdir, "imaging.csv"), "a", encoding="utf-8") as handle:
            if handle.tell() == 0:
                handle.write("RA,Dec,SNR,CENT_SNR,PC,DETECT_STATUS,dt,duration,e1,e2\n")
            if time_seed:
                dt_arr = np.array(dt_det)[sorted_indices].tolist()
                duration_arr = np.array(duration_det)[sorted_indices].tolist()
                for r, d, s1, s2, pc_val, status_val, t, dur, e1_val, e2_val in zip(ra, dec, snr, cent_snr, pc, det_status, dt_arr, duration_arr, e1, e2):
                    handle.write(f"{round(r,4)},{round(d,4)},{round(s1,4)},{round(s2,4)},{round(pc_val,4)},{status_val},{round(t,4)},{round(dur,4)},{round(e1_val,4)},{round(e2_val,4)}\n")
            else:
                t_bin_arr = np.array(t_bin)[sorted_indices]
                for r, d, s1, s2, pc_val, status_val, dur, e1_val, e2_val in zip(ra, dec, snr, cent_snr, pc, det_status, t_bin_arr, e1, e2):
                    handle.write(f"{round(r,4)},{round(d,4)},{round(s1,4)},{round(s2,4)},{round(pc_val,4)},{status_val},{round(min_time,4)},{round(dur,4)},{round(e1_val,4)},{round(e2_val,4)}\n")
        return True
    logging.info("No imaging sources found.")
    return False


def mosaic(t0, event, workdir, min_time, max_time, healpix_nside=1024, skyview_nprocs=10, mosaic_nprocs=10, loop=False):
    ra = []
    dec = []
    snr = []
    t_start = []
    t_end = []
    e1 = []
    e2 = []
    for energybins in [[15, 350] * u.keV]:
        dt_0 = max_time - min_time
        t_cent = min_time + dt_0 / 2.0
        max_dur = 16.384 if loop else dt_0
        while dt_0 <= max_dur:
            if len(np.arange(t_cent - dt_0 / 2, t_cent + dt_0 / 2, 0.2)) <= 2:
                start = t_cent - dt_0 / 2
                stop = t_cent + dt_0 / 2
                mid = (start + stop) / 2
                time_bins = np.array([start, mid, stop]) * u.s
            else:
                time_bins = np.arange(t_cent - dt_0 / 2, t_cent + dt_0 / 2, 0.2) * u.s
            logging.info(f"time bins: {np.round(time_bins.value, 3)}")
            slew_skyviews = ba.parallel.create_event_skyview(event, timebins=time_bins, energybins=energybins, is_relative=True, parse_images=False, T0=t0, nprocs=skyview_nprocs, input_dict=dict(aperture="CALDB:DETECTION"))
            valid_skyviews = []
            for idx, skyview in enumerate(slew_skyviews):
                try:
                    if not skyview.is_mosaic and skyview.sky_img is None:
                        skyview._parse_skyimages()

                    has_required_images = (
                        skyview.sky_img is not None
                        and skyview.pcode_img is not None
                        and skyview.bkg_stddev_img is not None
                    )
                    if has_required_images:
                        valid_skyviews.append(skyview)
                    else:
                        logging.warning(
                            "Skipping skyview %s due missing images (sky=%s, pcode=%s, bkg=%s)",
                            idx,
                            skyview.sky_img is not None,
                            skyview.pcode_img is not None,
                            skyview.bkg_stddev_img is not None,
                        )
                except Exception:
                    logging.warning(f"Skipping skyview {idx} due parse error: {traceback.format_exc()}")

            if len(valid_skyviews) == 0:
                logging.warning(
                    "No valid skyviews available for mosaic in interval [%.3f, %.3f]",
                    t_cent - dt_0 / 2,
                    t_cent + dt_0 / 2,
                )
                dt_0 *= 2
                continue

            if len(valid_skyviews) == 1:
                logging.warning(
                    "Only one valid skyview available in interval [%.3f, %.3f]; using single-skyview fallback",
                    t_cent - dt_0 / 2,
                    t_cent + dt_0 / 2,
                )
                mosaic_skyview = valid_skyviews[0]
            else:
                mosaic_skyview = ba.parallel.mosaic_skyview(valid_skyviews, healpix_nside=healpix_nside, nprocs=mosaic_nprocs)
            snr_contents = mosaic_skyview.snr_img.contents
            snr_flat = snr_contents[np.isfinite(snr_contents)]
            if snr_flat.size:
                max_snr = snr_flat.max()
                p99 = np.percentile(snr_flat, 99)
                p995 = np.percentile(snr_flat, 99.5)
                logging.info(f"Mosaic SNR stats: max={max_snr:.3f}, p99={p99:.3f}, p99.5={p995:.3f}")
            else:
                logging.info("Mosaic SNR stats: all values are non-finite")
            try:
                plt.close("all")
                plot_snr(mosaic_skyview, min_time, max_time, energybins[0].value, energybins[1].value, workdir, flag="mosaic")
            except Exception:
                logging.error(f"Error in plotting SNR: {traceback.format_exc()}")
            mosaic_detected_sources = mosaic_skyview.detect_sources(input_dict=dict(pcodethresh=0.0, snrthresh=5.5))
            log_mosaic(mosaic_detected_sources)
            if mosaic_detected_sources is not None:
                for item in mosaic_detected_sources:
                    if item["psffwhm_separation"] > 1:
                        logging.info(f"separation from close known source: {item['psffwhm_separation']}")
                        ra.append(item["SNR_skycoord"].ra.deg)
                        dec.append(item["SNR_skycoord"].dec.deg)
                        snr.append(item["SNR"])
                        t_start.append(t_cent - dt_0 / 2.0)
                        t_end.append(t_cent + dt_0 / 2.0)
                        e1.append(energybins[0].value)
                        e2.append(energybins[1].value)
            dt_0 *= 2
    if len(snr) > 0:
        sorted_indices = np.argsort(snr)[::-1]
        ra = [ra[n] for n in sorted_indices]
        dec = [dec[n] for n in sorted_indices]
        snr = [snr[n] for n in sorted_indices]
        t_start = [t_start[n] for n in sorted_indices]
        t_end = [t_end[n] for n in sorted_indices]
        e1 = [e1[n] for n in sorted_indices]
        e2 = [e2[n] for n in sorted_indices]
        with open(os.path.join(workdir, "mosaic.csv"), "a", encoding="utf-8") as handle:
            if handle.tell() == 0:
                handle.write("RA,Dec,SNR,t_start,t_end,e1,e2\n")
            for r, d, s, t1, t2, e_1, e_2 in zip(ra, dec, snr, t_start, t_end, e1, e2):
                handle.write(f"{round(float(r),4)},{round(float(d),4)},{round(float(s),4)},{round(float(t1),4)},{round(float(t2),4)},{round(float(e_1),4)},{round(float(e_2),4)}\n")
        return True
    return False


def read_results(event, t0, workdir):
    mosaic_csv = os.path.join(workdir, "mosaic.csv")
    if os.stat(mosaic_csv).st_size != 0:
        utils.sort_csv(workdir, "mosaic.csv")
        df_sorted = pd.read_csv(mosaic_csv).sort_values(by="SNR", ascending=False)
        ra_max = df_sorted.iloc[0]["RA"]
        dec_max = df_sorted.iloc[0]["Dec"]
        deltat_max = df_sorted.iloc[0]["t_end"] - df_sorted.iloc[0]["t_start"]
        logging.info(f"Highest SNR mosaic source: RA={ra_max}, Dec={dec_max}")
        df_sorted.to_csv(mosaic_csv, index=False)
        t_start = df_sorted.iloc[0]["t_start"]
        t_end = df_sorted.iloc[0]["t_end"]
        time_bins = np.arange(t_start, t_end, 0.2)
        if len(time_bins) == 2:
            time_bins = np.array([t_start, (t_start + t_end) / 2, t_end])
        time_bins = time_bins * u.s
        energybins = [df_sorted.iloc[0]["e1"], df_sorted.iloc[0]["e2"]] * u.keV
        map_mosaic(event, ra_max, dec_max, time_bins, energybins, t0, workdir)
        plot_lc(event, ra_max, dec_max, t0, deltat_max, workdir, pipe="mosaic")
        pc_time(ra_max, dec_max, event, t0, [time_bins[0], time_bins[-1]], workdir)
        sign = "+" if df_sorted.iloc[0]["t_start"] > 0 else "-"
        dur = df_sorted.iloc[0]["t_end"] - df_sorted.iloc[0]["t_start"]
        if utils.TRIG_INSTR and "IGWN" in utils.TRIG_INSTR:
            utils.post_telegram(os.path.join(workdir, "lc_mosaic.png"), f"Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}.\nSource found with mosaic at: RA={ra_max}, Dec={dec_max}, SNR={df_sorted.iloc[0]['SNR']}, at t0 {sign} {abs(df_sorted.iloc[0]['t_start'])}, duration {dur} s.")
            time.sleep(5)
            utils.post_telegram(os.path.join(workdir, "map_mosaic.png"), f"Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}.")
            utils.post_telegram(os.path.join(workdir, "mosaic.csv"), f"Tables for mosaic. Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}.")
        time.sleep(5)
        utils.post_slack(os.path.join(workdir, "lc_mosaic.png"), f"Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}.\nSource found with mosaic at: RA={ra_max}, Dec={dec_max}, SNR={df_sorted.iloc[0]['SNR']}, at t0 {sign} {abs(df_sorted.iloc[0]['t_start'])}, duration {dur} s.")
        utils.post_slack(os.path.join(workdir, "map_mosaic.png"), f"Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}.")
        utils.post_slack(os.path.join(workdir, "mosaic.csv"), f"Tables for mosaic. Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}.")
        utils.post_slack(os.path.join(workdir, "pc_time.png"), f"Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}")

    imaging_csv = os.path.join(workdir, "imaging.csv")
    if os.stat(imaging_csv).st_size != 0:
        utils.sort_csv(workdir, "imaging.csv")
        df_sorted = pd.read_csv(imaging_csv).sort_values(by="SNR", ascending=False)
        ra_max = df_sorted.iloc[0]["RA"]
        dec_max = df_sorted.iloc[0]["Dec"]
        deltat_max = df_sorted.iloc[0]["duration"]
        logging.info(f"Highest SNR imaging source: RA={ra_max}, Dec={dec_max}")
        df_sorted.to_csv(imaging_csv, index=False)
        plot_lc(event, ra_max, dec_max, t0, deltat_max, workdir, pipe="imaging")
        time_bins = np.array([df_sorted.iloc[0]["dt"], df_sorted.iloc[0]["duration"] + deltat_max]) * u.s
        energybins = [df_sorted.iloc[0]["e1"], df_sorted.iloc[0]["e2"]] * u.keV
        map_imaging(event, ra_max, dec_max, time_bins, energybins, t0, workdir)
        sign = "+" if df_sorted.iloc[0]["dt"] > 0 else "-"
        if utils.TRIG_INSTR and "IGWN" in utils.TRIG_INSTR:
            utils.post_telegram(os.path.join(workdir, "lc_imaging.png"), f"Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}.\nSource found with imaging at: RA={ra_max}, Dec={dec_max}, SNR={df_sorted.iloc[0]['SNR']}, at t0 {sign} {df_sorted.iloc[0]['dt']}, duration {df_sorted.iloc[0]['duration']} s.")
            time.sleep(5)
            utils.post_telegram(os.path.join(workdir, "map_imaging.png"), f"Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}.")
            utils.post_telegram(os.path.join(workdir, "imaging.csv"), f"Tables for imaging. Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}.")
        time.sleep(5)
        utils.post_slack(os.path.join(workdir, "lc_imaging.png"), f"Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}.\nSource found with imaging at: RA={ra_max}, Dec={dec_max}, SNR={df_sorted.iloc[0]['SNR']}, at t0 {sign} {abs(df_sorted.iloc[0]['dt'])}, duration {df_sorted.iloc[0]['duration']} s.")
        utils.post_slack(os.path.join(workdir, "map_imaging.png"), f"Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}.")
        utils.post_slack(os.path.join(workdir, "imaging.csv"), f"Tables for imaging. Trigger ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}.")


def dpi(event, t0, workdir):
    try:
        time_bins = [max(-50, event.data.time.min().value - t0), min(50, event.data.time.max().value - t0)] * u.s
        dpi_obj = event.create_dpi(timebins=time_bins, energybins=[15, 350] * u.keV, T0=t0, is_relative=True)
        dpi_obj.plot()
        plt.savefig(os.path.join(workdir, "dpi.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception:
        logging.error(f"Error in dpi: {traceback.format_exc()}")


def run_analysis(event, t0, workdir, tmin, tmax, pipe, local, energybins=[15.0, 350] * u.keV, healpix_nside=512, skyview_nprocs=10, mosaic_nprocs=10):
    try:
        img_found = False
        failure = False
        att_time = np.asarray(getattr(event.attitude.time, "value", event.attitude.time), dtype=float)
        data_time = np.asarray(getattr(event.data.time, "value", event.data.time), dtype=float)
        data_time_min = np.nanmin(data_time)
        data_time_max = np.nanmax(data_time)

        att_window_mask = (att_time > data_time_min) & (att_time < data_time_max)
        settled_flags = np.asarray(event.attitude.is_10arcmin_settled) & np.asarray(event.attitude.is_settled)
        slew_flags = (~np.asarray(event.attitude.is_10arcmin_settled)) | (~np.asarray(event.attitude.is_settled))

        for fname in ["imaging.csv", "mosaic.csv"]:
            with open(os.path.join(workdir, fname), "w", encoding="utf-8") as handle:
                handle.truncate(0)
        slew_idx = np.where(att_window_mask & slew_flags)
        settled_idx = np.where(att_window_mask & settled_flags)
        plt.plot(att_time - t0, event.attitude.ra, label="RA", color="red")
        plt.plot(att_time - t0, event.attitude.dec, label="Dec", color="blue")
        t_tot = 10
        if len(settled_idx[0]) > 0:
            settled_times = att_time[settled_idx]
            settled_mask = (settled_times >= t0 - 5) & (settled_times <= t0 + t_tot)
            min_settled = settled_times[settled_mask].min() if any(settled_mask) else -np.inf
            max_settled = settled_times[settled_mask].max() if any(settled_mask) else -np.inf
        else:
            min_settled = -np.inf
            max_settled = -np.inf
        if len(slew_idx[0]) > 0:
            slew_times = att_time[slew_idx]
            slew_mask = (slew_times >= t0 - 5) & (slew_times <= t0 + t_tot)
            min_slew = slew_times[slew_mask].min() if any(slew_mask) else -np.inf
            max_slew = slew_times[slew_mask].max() if any(slew_mask) else -np.inf
        else:
            min_slew = -np.inf
            max_slew = -np.inf
        logging.info(f"min slew: {min_slew - t0}, max slew {max_slew - t0}")
        logging.info(f"min settled: {min_settled - t0}, max settled {max_settled - t0}")
        ra_diff = np.diff(event.attitude.ra.value)
        dec_diff = np.diff(event.attitude.dec.value)
        margin = 1 / 60
        change_indices = np.where((np.abs(ra_diff) > margin) | (np.abs(dec_diff) > margin))[0]
        slew_intervals = []
        for idx in change_indices:
            plt.axvspan(att_time[idx] - t0, att_time[idx + 1] - t0, color="gray", alpha=0.3, linewidth=0, label="Slew" if idx == change_indices[0] else None)
            slew_intervals.append((att_time[idx] - t0, att_time[idx + 1] - t0))
        slew_intervals = np.array(slew_intervals)
        if max(att_time - t0) < 0:
            slew_intervals = np.array([[max(att_time) - t0, 150]])
        plt.xlim(-50, 50)
        plt.xlabel("Time [s] (t - t0)")
        plt.ylabel("RA/Dec [deg]")
        plt.legend()
        plt.savefig(os.path.join(workdir, "attitude.png"), dpi=500)
        plt.close()
        dpi(event, t0, workdir)
        plot_ext_bat_diagnostic(event, t0, workdir)
        try:
            logging.info(f"attitude interval: {att_time.min() - t0}, {att_time.max() - t0}")
            logging.info(f"event interval: {data_time.min() - t0}, {data_time.max() - t0}")
            logging.info(f"settled interval: {att_time[settled_idx].min() - t0}, {att_time[settled_idx].max() - t0}")
            logging.info(f"slew interval: {att_time[slew_idx].min() - t0}, {att_time[slew_idx].max() - t0}")
        except Exception:
            logging.error("error in print statements")
        if tmin is not None and tmax is not None and pipe is not None:
            if pipe == "imaging":
                logging.info(f"doing ad-hoc imaging in the time interval [{tmin},{tmax}]")
                try:
                    img_found = img_found or imaging(t0, event, workdir, float(tmin), float(tmax), None, False)
                except Exception:
                    logging.error(f"error in imaging: {traceback.format_exc()}")
                    failure = True
            elif pipe == "mosaic":
                logging.info(f"doing ad-hoc mosaic in the time interval [{tmin},{tmax}]")
                try:
                    img_found = img_found or mosaic(t0, event, workdir, float(tmin), float(tmax), loop=False)
                except Exception:
                    logging.error(f"error in mosaic: {traceback.format_exc()}")
                    failure = True
            else:
                logging.error("pipe must be either imaging or mosaic")
            max_snr, im, max_snr_imaging, mos, max_snr_mosaic = utils.read_snr(workdir)
            if max_snr is not None and max_snr >= 6:
                if im:
                    utils.sort_csv(workdir, "imaging.csv")
                    utils.post_slack(os.path.join(workdir, "imaging.csv"), f"Imaging found results with max SNR {max_snr_imaging} in the time interval [{tmin}, {tmax}]")
                    if utils.TRIG_INSTR and "IGWN" in utils.TRIG_INSTR:
                        utils.post_telegram(os.path.join(workdir, "imaging.csv"), f"Imaging found results with max SNR {max_snr_imaging} in the time interval [{tmin}, {tmax}]")
                if mos:
                    utils.sort_csv(workdir, "mosaic.csv")
                    utils.post_slack(os.path.join(workdir, "mosaic.csv"), f"Mosaic found results with max SNR {max_snr_mosaic} in the time interval [{tmin}, {tmax}]")
                    if utils.TRIG_INSTR and "IGWN" in utils.TRIG_INSTR:
                        utils.post_telegram(os.path.join(workdir, "mosaic.csv"), f"Mosaic found results with max SNR {max_snr_mosaic} in the time interval [{tmin}, {tmax}]")
            read_results(event, t0, workdir)
            logging.info(f"Failure: {failure}")
            return failure
        else:
            time_seed_file = os.path.join(workdir, "time_seeds.csv")
            if os.path.exists(time_seed_file):
                try:
                    time_seed_df = pd.read_csv(time_seed_file).sort_values(by="snr", ascending=False)
                except Exception:
                    time_seed_df = pd.DataFrame()
                if not time_seed_df.empty:
                    dt = time_seed_df["dt"][: min(10, len(time_seed_df))].to_numpy()
                    duration = time_seed_df["duration"][: min(10, len(time_seed_df))].to_numpy()
                    logging.info("doing imaging using the time seeds by NITRATES")
                    try:
                        img_found = img_found or imaging(t0, event, workdir, None, None, [dt, duration], True)
                        logging.info(f"condition img_found {img_found}")
                        failure = False
                    except Exception:
                        logging.error(f"error in imaging: {traceback.format_exc()}")
                        failure = True
                else:
                    logging.info("time_seed.csv is empty")
            else:
                logging.info("time_seed.csv not found in workdir")
            max_snr, im, max_snr_imaging, mos, max_snr_mosaic = utils.read_snr(workdir)
            if max_snr is not None and max_snr >= 6 and im:
                utils.sort_csv(workdir, "imaging.csv")
                utils.post_slack(os.path.join(workdir, "imaging.csv"), f"Imaging with NITRATES seeds found results with max SNR {max_snr_imaging}, trig ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}")
                if utils.TRIG_INSTR and "IGWN" in utils.TRIG_INSTR:
                    utils.post_telegram(os.path.join(workdir, "imaging.csv"), f"Imaging with NITRATES seeds found results with max SNR {max_snr_imaging}, trig ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}")
            seed_max, dur_max, seeds = utils.cust_seeds(t0, workdir)
            if seed_max is not None:
                logging.info(f"Found seed with custom search at {round(seed_max, 3)} s with duration {dur_max} ms")
                try:
                    filename = utils.resolve_event_file(workdir)
                    if filename is None:
                        logging.error("No event file found for refined custom seed search; skipping.")
                    else:
                        if not filename.endswith("filter_evdata.fits"):
                            logging.info(f"Using fallback event file for refined custom seeds: {filename}")
                        event_times, gti_start, gti_stop = utils.load_event_times(filename)
                        bin_centers, counts = utils.bin_light_curve(event_times, 0.016, gti_start, gti_stop, t0)
                        best_params, samples, norm_val = utils.fit_background_linear(bin_centers, counts, t0)
                        model_bkg = utils.model(bin_centers, t0, best_params, norm_val)
                        counts_sub = counts - model_bkg
                        slew_times = att_time[slew_idx] - t0
                        for item in seeds:
                            max_snr = utils.read_snr(workdir)[0]
                            if max_snr is not None and max_snr >= 20:
                                logging.info(f"Maximum SNR found: {max_snr}, skipping refined seed search")
                                break
                            if item[0] is not None and -20 < item[0] < 20:
                                t_cent = item[0]
                                dt = item[1] / 1000.0
                                t_cent, dt = utils.refined_seed_search(bin_centers, counts_sub, t_cent, dt, n_trials=10000)
                                dt = min(dt, 10)
                                logging.info(f"Refined seed found at {round(t_cent, 3)} s with duration {round(dt, 3)} s")
                                interval_start = t_cent - dt / 2
                                interval_end = t_cent + dt / 2
                                in_interval = np.any([(interval_start <= slew_end and interval_end >= slew_start) for slew_start, slew_end in slew_intervals])
                                logging.info(f"Interval [{interval_start:.3f}, {interval_end:.3f}] intersects with any slew interval: {in_interval}")
                                if dt <= 1.024 / 4:
                                    try:
                                        result = imaging(t0, event, workdir, None, None, [[t_cent - dt / 2], [dt]], True)
                                        img_found = img_found or result
                                    except Exception:
                                        logging.error(f"error in imaging: {traceback.format_exc()}")
                                        failure = True
                                if 1.024 / 4 <= dt < 15:
                                    try:
                                        if in_interval:
                                            result = mosaic(t0, event, workdir, t_cent - dt / 2, t_cent + dt / 2)
                                            img_found = img_found or result
                                        else:
                                            result = imaging(t0, event, workdir, None, None, [[t_cent - dt / 2], [dt]], True)
                                            img_found = img_found or result
                                    except Exception:
                                        logging.error(f"error in mosaic: {traceback.format_exc()}")
                                        failure = True
                    max_snr, im, max_snr_imaging, mos, max_snr_mosaic = utils.read_snr(workdir)
                    logging.info(f"Maximum SNR after custom seeds search: {max_snr}")
                    if max_snr is not None and max_snr >= 6:
                        if im:
                            utils.sort_csv(workdir, "imaging.csv")
                            utils.post_slack(os.path.join(workdir, "imaging.csv"), f"Imaging with custom seeds search found results with max SNR {max_snr_imaging}, trig ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}")
                            if utils.TRIG_INSTR and "IGWN" in utils.TRIG_INSTR:
                                utils.post_telegram(os.path.join(workdir, "imaging.csv"), f"Imaging with custom seeds search found results with max SNR {max_snr_imaging}, trig ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}")
                        if mos:
                            utils.sort_csv(workdir, "mosaic.csv")
                            utils.post_slack(os.path.join(workdir, "mosaic.csv"), f"Mosaic with custom seeds search found results with max SNR {max_snr_mosaic}, trig ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}")
                            if utils.TRIG_INSTR and "IGWN" in utils.TRIG_INSTR:
                                utils.post_telegram(os.path.join(workdir, "mosaic.csv"), f"Mosaic with custom seeds search found results with max SNR {max_snr_mosaic}, trig ID {utils.TRIGRID}, external trigger {utils.EXT_TRIG}")
                    failure = False
                except Exception:
                    logging.error(f"error: {traceback.format_exc()}")
                    failure = True
            else:
                logging.info("No custom seeds found")
        read_results(event, t0, workdir)
        logging.info(f"Failure: {failure}")
    except Exception:
        logging.error(f"Error in analysis: {traceback.format_exc()}")
        failure = True
    return failure


def _normalize_obsid(obsid):
    obsid_str = str(obsid)
    if obsid_str.isdigit() and len(obsid_str) < 11:
        obsid_str = obsid_str.zfill(11)
    return obsid_str


def _parse_trigger_time(triggertime):
    time_str = triggertime[:-1] if triggertime.endswith("Z") else triggertime
    return Time(time_str, format="isot", scale="utc")


def _find_obsid_attitude_pat(download_root, obsid):
    obsid = _normalize_obsid(obsid)
    aux_dir = Path(download_root) / obsid / "auxil"
    if not aux_dir.exists():
        return None
    for pattern in (f"sw{obsid}pat.fits.gz", f"sw{obsid}pat.fits"):
        matches = sorted(aux_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_obsid_attitude_sat_or_mkf(download_root, obsid):
    obsid = _normalize_obsid(obsid)
    aux_dir = Path(download_root) / obsid / "auxil"
    if not aux_dir.exists():
        return None
    patterns = (
        f"sw{obsid}sat.fits.gz",
        f"sw{obsid}sat.fits",
        f"sw{obsid}s.mkf.gz",
        f"sw{obsid}s.mkf",
        f"sw{obsid}x.mkf.gz",
        f"sw{obsid}x.mkf",
    )
    for pattern in patterns:
        matches = sorted(aux_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _project_root():
    here = Path(__file__).resolve().parent
    if here.name == "batglimpse":
        return here.parent
    return here


def _has_required_guano_subdirs(obsid_dir):
    return (
        obsid_dir.joinpath("bat", "event").is_dir()
        and obsid_dir.joinpath("bat", "hk").is_dir()
        and obsid_dir.joinpath("auxil").is_dir()
    )


def _hydrate_obsid_from_local_cache(download_root, obsid):
    target_obsid_dir = Path(download_root) / _normalize_obsid(obsid)
    if _has_required_guano_subdirs(target_obsid_dir):
        return True

    search_root = _project_root()
    for candidate in sorted(search_root.glob(f"**/bat_downloads/{_normalize_obsid(obsid)}")):
        try:
            if candidate.resolve() == target_obsid_dir.resolve():
                continue
        except FileNotFoundError:
            continue
        if not _has_required_guano_subdirs(candidate):
            continue
        target_obsid_dir.mkdir(parents=True, exist_ok=True)
        for rel_dir in ("auxil", "bat/event", "bat/hk", "bat/rate"):
            src = candidate / rel_dir
            if src.exists():
                shutil.copytree(src, target_obsid_dir / rel_dir, dirs_exist_ok=True)
        logging.info("Hydrated obsid %s data from local cache at %s", obsid, candidate)
        return _has_required_guano_subdirs(target_obsid_dir)

    return False


def _has_acs_data_extension(path):
    path = Path(path)
    if not path.exists():
        return False
    with fits.open(path) as hdul:
        return "ACS_DATA" in hdul


def _has_valid_attitude_time_units(path):
    path = Path(path)
    if not path.exists():
        return False
    try:
        with fits.open(path) as hdul:
            if "ACS_DATA" not in hdul:
                return False
            columns = hdul["ACS_DATA"].columns
            if "TIME" not in columns.names:
                return False
            time_unit = columns["TIME"].unit
            if time_unit is None or str(time_unit).strip() == "":
                return False
            return u.Unit(time_unit).is_equivalent(u.s)
    except Exception:
        return False


def _read_time_values(path):
    with fits.open(path) as hdul:
        if "ACS_DATA" in hdul and "TIME" in hdul["ACS_DATA"].columns.names:
            return np.asarray(hdul["ACS_DATA"].data["TIME"], dtype=float)
        for hdu in hdul[1:]:
            names = getattr(getattr(hdu, "data", None), "names", None)
            if names is not None and "TIME" in names:
                return np.asarray(hdu.data["TIME"], dtype=float)
    raise ValueError(f"No TIME column found in {path}")


def _attitude_covers_t0(path, t0_met, tmin=None, tmax=None):
    try:
        times = _read_time_values(path)
    except Exception:
        return False

    times = times[np.isfinite(times)]
    if times.size == 0:
        return False

    if tmin is not None and tmax is not None:
        w_start = t0_met + float(min(tmin, tmax))
        w_stop = t0_met + float(max(tmin, tmax))
    else:
        w_start = t0_met - 2.0
        w_stop = t0_met + 5.0

    in_window = np.any((times >= w_start) & (times <= w_stop))
    has_bracket = np.any(times <= t0_met) and np.any(times >= t0_met)
    return bool(in_window and has_bracket)


def _is_batanalysis_attitude_file(path):
    path = Path(path)
    name = path.name.lower()
    if "sat.fits" in name:
        return True
    if name.endswith(".mkf") or name.endswith(".mkf.gz"):
        return True
    suffixes = [suffix.lower() for suffix in path.suffixes]
    return ".sat" in suffixes or ".mkf" in suffixes


def _is_blank_or_dimensionless_unit(unit):
    if unit is None:
        return True
    try:
        parsed = u.Unit(unit)
    except Exception:
        return str(unit).strip() == ""
    return parsed == u.dimensionless_unscaled


def _normalize_tables_for_stack(table1, table2):
    normalized1 = table1.copy(copy_data=True)
    normalized2 = table2.copy(copy_data=True)

    for colname in normalized1.colnames:
        if colname not in normalized2.colnames:
            continue

        col1 = normalized1[colname]
        col2 = normalized2[colname]
        unit1 = getattr(col1, "unit", None)
        unit2 = getattr(col2, "unit", None)

        if unit1 == unit2:
            continue

        if _is_blank_or_dimensionless_unit(unit1) and not _is_blank_or_dimensionless_unit(unit2):
            normalized1[colname] = np.asarray(col1)
            normalized1[colname].unit = unit2
            continue

        if _is_blank_or_dimensionless_unit(unit2) and not _is_blank_or_dimensionless_unit(unit1):
            normalized2[colname] = np.asarray(col2)
            normalized2[colname].unit = unit1
            continue

        if not _is_blank_or_dimensionless_unit(unit1) and not _is_blank_or_dimensionless_unit(unit2):
            try:
                normalized2[colname] = col2.to(unit1)
                continue
            except (u.UnitConversionError, AttributeError):
                raise ValueError(f"Column {colname!r} has incompatible units: {unit1} vs {unit2}")

    return normalized1, normalized2


def _merge_two_attitude_tables(table1, table2):
    table1, table2 = _normalize_tables_for_stack(table1, table2)
    combined = vstack([table1, table2], metadata_conflicts="silent", join_type="exact")
    if "TIME" in combined.colnames:
        if hasattr(combined, "unique"):
            merged = combined.unique(keys=["TIME"])
        else:
            merged = unique(combined, keys=["TIME"])
        merged.sort("TIME")
        return merged
    return combined


def _copy_non_structural_header_cards(src_hdu, dst_hdu):
    skip_prefixes = (
        "TFORM",
        "TTYPE",
        "TUNIT",
        "TDIM",
        "TDISP",
        "TNULL",
        "TSCAL",
        "TZERO",
        "TBCOL",
    )
    skip_keys = {
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "EXTNAME",
        "CHECKSUM",
        "DATASUM",
    }
    for card in src_hdu.header.cards:
        keyword = card.keyword
        if keyword in skip_keys or keyword.startswith(skip_prefixes):
            continue
        if keyword == "COMMENT":
            dst_hdu.header.add_comment(card.value)
            continue
        if keyword == "HISTORY":
            dst_hdu.header.add_history(card.value)
            continue
        if keyword not in dst_hdu.header:
            dst_hdu.header[keyword] = (card.value, card.comment)


def _merge_attitude_pat_files(file1, file2, output):
    file1 = Path(file1)
    file2 = Path(file2)
    with fits.open(file1) as hdul1, fits.open(file2) as hdul2:
        primary = fits.PrimaryHDU(
            data=None if hdul1[0].data is None else np.array(hdul1[0].data, copy=True),
            header=hdul1[0].header.copy(),
        )
        merged_hdus = [primary]

        merged_any = False
        for ext_name in ("ATTITUDE", "ACS_DATA"):
            if ext_name not in hdul1 or ext_name not in hdul2:
                continue
            t1 = Table.read(file1, hdu=ext_name)
            t2 = Table.read(file2, hdu=ext_name)
            merged_table = _merge_two_attitude_tables(t1, t2)
            merged_hdu = fits.table_to_hdu(merged_table)
            merged_hdu.name = ext_name
            _copy_non_structural_header_cards(hdul1[ext_name], merged_hdu)
            merged_hdus.append(merged_hdu)
            merged_any = True

        if not merged_any:
            raise ValueError(f"No common ATTITUDE/ACS_DATA extensions to merge: {file1} and {file2}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(merged_hdus).writeto(output, overwrite=True)
    return output


def _get_ordered_obsids_from_obsquery(triggertime):
    trig = _parse_trigger_time(triggertime)
    begin = (trig - TimeDelta(60, format="sec")).datetime
    end = (trig + TimeDelta(3 * 3600, format="sec")).datetime
    query = ObsQuery(begin=begin, end=end)

    ordered = []
    seen = set()
    for row in query:
        obsid = getattr(row, "obsid", None)
        if obsid is None:
            continue
        obsid = _normalize_obsid(obsid)
        if obsid not in seen:
            seen.add(obsid)
            ordered.append(obsid)
    return ordered


def _find_next_obsid(triggertime, current_obsid):
    current_obsid = _normalize_obsid(current_obsid)
    ordered_obsids = _get_ordered_obsids_from_obsquery(triggertime)
    if current_obsid in ordered_obsids:
        idx = ordered_obsids.index(current_obsid)
        if idx + 1 < len(ordered_obsids):
            return ordered_obsids[idx + 1]
    for obsid in ordered_obsids:
        if obsid != current_obsid:
            return obsid
    return None


def _ensure_attitude_file_for_t0(obsid, triggertime, t0_met, workdir, download_root, tmin, tmax):
    obsid = _normalize_obsid(obsid)
    workdir = Path(workdir)
    download_root = Path(download_root)

    local_sat = workdir / "attitude.sat"
    local_fits = workdir / "attitude.fits"
    current_sat_or_mkf = _find_obsid_attitude_sat_or_mkf(download_root, obsid)
    current_pat = _find_obsid_attitude_pat(download_root, obsid)

    candidates = [local_sat, local_fits]
    if current_sat_or_mkf is not None:
        candidates.append(current_sat_or_mkf)
    if current_pat is not None:
        candidates.append(current_pat)

    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        if not _has_acs_data_extension(candidate):
            logging.warning(f"Attitude candidate {candidate} has no ACS_DATA extension; skipping")
            continue
        if not _has_valid_attitude_time_units(candidate):
            logging.warning(f"Attitude candidate {candidate} has invalid/missing TIME units in ACS_DATA; skipping")
            continue
        if _attitude_covers_t0(candidate, t0_met, tmin=tmin, tmax=tmax):
            logging.info(f"Using attitude file with t0 coverage: {candidate}")
            return candidate

    if current_pat is None and current_sat_or_mkf is None:
        logging.info(f"Downloading auxil for obsid {obsid} to recover attitude PAT file")
        try:
            data = Data(obsid=obsid, auxil=True, outdir=str(download_root), clobber=True, uksdc=True)
            logging.info(data)
        except Exception:
            logging.error(f"Error downloading auxil for obsid {obsid}: {traceback.format_exc()}")
        current_sat_or_mkf = _find_obsid_attitude_sat_or_mkf(download_root, obsid)
        current_pat = _find_obsid_attitude_pat(download_root, obsid)

    if current_pat is None:
        if current_sat_or_mkf is not None:
            logging.warning(
                "No PAT attitude file found for obsid %s; falling back to SAT/MKF file %s",
                obsid,
                current_sat_or_mkf,
            )
            return current_sat_or_mkf
        raise RuntimeError(f"No PAT attitude file found for obsid {obsid}; cannot build t0-covered attitude")

    next_obsid = _find_next_obsid(triggertime, obsid)
    if next_obsid is None:
        logging.warning("Could not find next obsid from ObsQuery; falling back to current PAT attitude")
        return current_pat

    next_pat = _find_obsid_attitude_pat(download_root, next_obsid)
    if next_pat is None:
        logging.info(f"Downloading auxil for next obsid {next_obsid} for attitude merge")
        try:
            data = Data(obsid=next_obsid, auxil=True, outdir=str(download_root), clobber=True, uksdc=True)
            logging.info(data)
        except Exception:
            logging.error(f"Error downloading auxil for next obsid {next_obsid}: {traceback.format_exc()}")
        next_pat = _find_obsid_attitude_pat(download_root, next_obsid)

    if next_pat is None:
        logging.warning(f"Could not find next obsid PAT file for {next_obsid}; falling back to current PAT attitude")
        return current_pat

    merged_attitude = workdir / "attitude.sat"
    merged_path = _merge_attitude_pat_files(current_pat, next_pat, merged_attitude)
    merged_has_acs = _has_acs_data_extension(merged_path)
    merged_has_time_units = _has_valid_attitude_time_units(merged_path)
    merged_covers = _attitude_covers_t0(merged_path, t0_met, tmin=tmin, tmax=tmax)
    logging.info(
        "Merged attitude created from obsids %s and %s at %s (ACS_DATA=%s, TIME_units_ok=%s, covers_t0=%s)",
        obsid,
        next_obsid,
        merged_path,
        merged_has_acs,
        merged_has_time_units,
        merged_covers,
    )
    if not merged_has_acs:
        raise RuntimeError(f"Merged attitude file {merged_path} does not contain ACS_DATA")
    if not merged_has_time_units:
        raise RuntimeError(f"Merged attitude file {merged_path} has invalid TIME units in ACS_DATA")
    if not merged_covers:
        logging.warning(f"Merged attitude file {merged_path} still does not cover t0 window")
    return merged_path


def guano_query(triggertime, ext_obsid, workdir, tmin, tmax, pipe, healpix_nside, skyview_nprocs, mosaic_nprocs):
    if "." not in triggertime.split("T")[1]:
        triggertime = triggertime + ".000Z"
    else:
        triggertime = triggertime + "Z"
    logging.info(f"Using triggertime: {triggertime}")
    guano = GUANO(triggertime=triggertime, username="echolocation", shared_secret="TqdA2N8iD0KBMZpIgFxU", subthreshold=True, successful=False)
    logging.info(guano)
    download_root = os.path.join(workdir, "bat_downloads")
    os.makedirs(download_root, exist_ok=True)
    for item in guano:
        logging.info(f"running {item}")
        if item.data.exposure is None:
            logging.error(f"No exposure time found for obsid {item.obsid}. Skipping this obsid.")
            return True
        obsid = _normalize_obsid(ext_obsid if ext_obsid is not None else item.obsid)
        t0_met = item.triggertime.met
        obsid_dir = Path(download_root) / obsid
        detmask_path = Path(workdir) / "detmask.fits"
        event_path = Path(workdir) / "filter_evdata.fits"
        attitude_sat_path = Path(workdir) / "attitude.sat"
        attitude_fits_path = Path(workdir) / "attitude.fits"
        have_local_inputs = event_path.exists() and detmask_path.exists() and (attitude_sat_path.exists() or attitude_fits_path.exists())
        start_time_try = time.time()
        event = None
        retry_count = 0
        while time.time() - start_time_try < GUANO_DATA_MAX_WAIT_SECONDS and event is None:
            try:
                have_obsid_data = _has_required_guano_subdirs(obsid_dir)
                if not have_obsid_data:
                    have_obsid_data = _hydrate_obsid_from_local_cache(download_root, obsid)
                if not have_obsid_data:
                    data = Data(obsid=obsid, bat=True, outdir=download_root, clobber=True, uksdc=True)
                    logging.info(data)
                else:
                    logging.info(f"Using existing data for obsid {obsid}; skipping download.")
                ba.datadir(download_root)
                event = ba.BatEvent(obsid, is_guano=True)
                if detmask_path.exists():
                    event.detector_quality_file = detmask_path
                if event_path.exists():
                    event.event_files = event_path

                selected_attitude_file = _ensure_attitude_file_for_t0(
                    obsid=obsid,
                    triggertime=triggertime,
                    t0_met=t0_met,
                    workdir=workdir,
                    download_root=download_root,
                    tmin=tmin,
                    tmax=tmax,
                )
                if selected_attitude_file is not None:
                    event.attitude_file = Path(selected_attitude_file)
                    if _has_acs_data_extension(event.attitude_file) and _is_batanalysis_attitude_file(event.attitude_file):
                        event.attitude = ba.Attitude.from_file(event.attitude_file)
                    elif not _has_acs_data_extension(event.attitude_file):
                        raise RuntimeError(f"Selected attitude file has no ACS_DATA extension: {event.attitude_file}")
                    else:
                        logging.info(
                            "Selected attitude file %s is not a .sat/.mkf file; skipping Attitude.from_file and using event parser",
                            event.attitude_file,
                        )
                elif attitude_sat_path.exists():
                    event.attitude_file = attitude_sat_path
                elif attitude_fits_path.exists():
                    event.attitude_file = attitude_fits_path
                event._parse_event_file()
            except Exception:
                event = None
                retry_count += 1
                logging.error(f"Failed to create Data and BatEvent for obsid {obsid}: {traceback.format_exc()}")
                if retry_count >= GUANO_MAX_RETRIES:
                    logging.error(
                        "Reached max GUANO retries for obsid %s (%d/%d)",
                        obsid,
                        retry_count,
                        GUANO_MAX_RETRIES,
                    )
                    break
                elapsed = time.time() - start_time_try
                remaining = max(0.0, GUANO_DATA_MAX_WAIT_SECONDS - elapsed)
                if remaining <= 0:
                    break
                sleep_seconds = min(float(GUANO_RETRY_SLEEP_SECONDS), remaining)
                logging.info(
                    "Retrying GUANO fetch for obsid %s in %.0f s (attempt %d/%d, elapsed %.0f/%.0f s)",
                    obsid,
                    sleep_seconds,
                    retry_count + 1,
                    GUANO_MAX_RETRIES,
                    elapsed,
                    float(GUANO_DATA_MAX_WAIT_SECONDS),
                )
                time.sleep(sleep_seconds)
        if event is None:
            logging.error(
                "Timed out waiting for GUANO data for obsid %s after %d s (retries=%d)",
                obsid,
                GUANO_DATA_MAX_WAIT_SECONDS,
                retry_count,
            )
            return True
        logging.info(f"obsid: {obsid}, triggertime: {triggertime}, t0_met: {t0_met}")

        db_file = Path(workdir) / "results.db"

        if not db_file.exists():

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nitrates.data_prep.mkdb",
                    "--work_dir",
                    workdir,
                ],
                check=True,
                capture_output=True,
                text=True
            )   

        subprocess.run(
            [
                sys.executable,
                "-m",
                "nitrates.data_prep.do_data_setup",
                "--work_dir",
                workdir,
                "--trig_time",
                triggertime[:-1],
                "--Obsid_Dir",
                str(obsid_dir),
            ],
            # check=True,
            capture_output=True,
            text=True
        )

        return run_analysis(event, t0_met, workdir, tmin, tmax, pipe, local=False, healpix_nside=healpix_nside, skyview_nprocs=skyview_nprocs, mosaic_nprocs=mosaic_nprocs)
