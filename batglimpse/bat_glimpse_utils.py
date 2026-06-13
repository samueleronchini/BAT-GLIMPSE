import logging
import multiprocessing
import os
import random
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path

import emcee
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.time import Time, TimeDelta
from ligo.gracedb.rest import GraceDb
from scipy.optimize import curve_fit
from scipy.special import gammaln
from tqdm import tqdm

import batanalysis as ba

try:
    from gdt.missions.fermi.gbm.finders import TriggerFinder
except ModuleNotFoundError:
    TriggerFinder = None

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ModuleNotFoundError:
    WebClient = None
    SlackApiError = Exception

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")
GRACE_DB = GraceDb()
SLACK_CLIENT = WebClient(token=SLACK_BOT_TOKEN) if WebClient is not None and SLACK_BOT_TOKEN else None

WORKDIR = None
TRIGRID = None
EXT_TRIG = None
TRIG_INSTR = None


def set_runtime_context(*, workdir, trigid, ext_trig, trig_instr):
    global WORKDIR, TRIGRID, EXT_TRIG, TRIG_INSTR
    WORKDIR = workdir
    TRIGRID = trigid
    EXT_TRIG = ext_trig
    TRIG_INSTR = trig_instr


def _fermi_map_files(workdir):
    return {
        fname
        for fname in os.listdir(workdir)
        if (fname.startswith("glg_healpix") or fname.startswith("ext_loc_fermi_glg_healpix"))
        and fname.endswith((".fit", ".fits", ".fit.gz", ".fits.gz"))
    }


def download_fermi_map(trigger_time, workdir):
    if TriggerFinder is None:
        logging.warning("gdt.missions.fermi is not available; skipping Fermi map download.")
        return

    dt = datetime.strptime(trigger_time[:19], "%Y-%m-%dT%H:%M:%S")
    year = dt.year % 100
    month = dt.month
    day = dt.day
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds = round((dt - midnight).total_seconds() / (24 * 3600) * 1000)
    fermi_trigid = f"{year:02d}{month:02d}{day:02d}{seconds:03d}"

    try:
        existing_files = _fermi_map_files(workdir)
        finder = TriggerFinder(fermi_trigid)
        logging.info("Searching Fermi archive path %s", finder.cwd)
        finder.get_trigdat(workdir, verbose=False)
        finder.get_healpix(workdir, verbose=False)

        downloaded_files = _fermi_map_files(workdir)
        if not downloaded_files:
            logging.info("No Fermi map found in Fermi archive")
            return

        candidate_pool = set(downloaded_files) - set(existing_files)
        if not candidate_pool:
            candidate_pool = downloaded_files
        export_file_name = max(candidate_pool, key=lambda fname: os.path.getmtime(os.path.join(workdir, fname)))
        old_path = os.path.join(workdir, export_file_name)

        if export_file_name.startswith("glg_"):
            new_path = os.path.join(workdir, export_file_name.replace("glg_", "ext_loc_fermi_glg_", 1))
            os.replace(old_path, new_path)
        else:
            new_path = old_path

        logging.info("Fermi map found: %s", os.path.basename(new_path))
    except Exception as exc:
        logging.error(f"Error downloading Fermi map: {exc}")
        traceback.print_exc()


def search_ext_maps(trigger_time, workdir):
    logging.info("searching map")
    logging.info("searching Fermi map")
    download_fermi_map(trigger_time, workdir)

    if TRIG_INSTR and "IGWN" in TRIG_INSTR:
        gw_id = next((x for x in EXT_TRIG if x.startswith("S")), next((x for x in EXT_TRIG if x.startswith("G")), None))
        x = GRACE_DB.files(gw_id).json()
        fits_files = [(name, url) for name, url in x.items() if "Bilby" in name and "fits" in name and "," not in name]
        if not fits_files:
            fits_files = [(name, url) for name, url in x.items() if "multiorder" in name and "fits" in name and "," not in name]
        if not fits_files:
            fits_files = [(name, url) for name, url in x.items() if "fits" in name and "," not in name]
        if fits_files:
            filename, url = fits_files[-1]
            response_ = requests.get(url)
            outname = os.path.join(workdir, f"ext_loc_{filename}")
            if response_.ok:
                with open(outname, "wb") as handle:
                    handle.write(response_.content)
                print(f"Downloaded: {outname}")


def post_telegram(file, message):
    if not BOT_TOKEN or not CHAT_ID:
        logging.warning("Telegram credentials are not configured; skipping Telegram upload.")
        return
    try:
        with open(file, "rb") as handle:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                data={"chat_id": CHAT_ID, "caption": message, "parse_mode": "Markdown"},
                files={"document": handle},
            )
    except Exception as exc:
        logging.error(f"Error posting to Telegram: {exc}")
        traceback.print_exc()


def post_slack(file, message):
    if SLACK_CLIENT is None or not CHANNEL_ID:
        logging.warning("Slack credentials are not configured; skipping Slack upload.")
        return
    try:
        with open(file, "rb") as file_content:
            response = SLACK_CLIENT.files_upload_v2(
                channel=CHANNEL_ID,
                file=file_content,
                title=os.path.basename(file),
                initial_comment=message,
            )
        print("✅ File uploaded:", response["file"]["id"])
    except SlackApiError as exc:
        logging.error(f"Error posting to Slack: {exc.response['error']}")
        traceback.print_exc()


def alert(*args):
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        logging.warning("Pushover credentials are not configured; skipping alert.")
        return
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={"token": PUSHOVER_TOKEN, "user": PUSHOVER_USER, "message": "\n".join(args)},
    )


def get_conn(db_fname):
    return sqlite3.connect(db_fname)


def query_data_utcslice(conn, utc0, utc1, table_name="SwiftQLevent"):
    sql = """SELECT * FROM %s
        Where UTCstart < '%s'
        and UTCstop > '%s' """ % (table_name, utc0, utc1)
    return pd.read_sql(sql, conn)


def get_obsid(t0):
    conn = get_conn("/storage/group/jak51/default/realtime_workdir/BATQL.db")
    apy_trig_time = Time(t0, format="isot")
    t_buff = TimeDelta(60.0, format="sec")
    t_bounds = (apy_trig_time + t_buff, apy_trig_time - t_buff)
    ev_data_table = query_data_utcslice(conn, t_bounds[0], t_bounds[1])
    if ev_data_table.empty:
        logging.info("No obsid found in the given time range.")
        return None
    return ev_data_table["obsid"].tolist()[-1]


def load_event_times(filename):
    with fits.open(filename) as hdul:
        event_times = None
        gti_start = None
        gti_stop = None
        for hdu in hdul:
            if hdu.name == "EVENTS":
                event_times = hdu.data["TIME"]
            elif hdu.name == "GTI":
                gti_start = hdu.data["START"]
                gti_stop = hdu.data["STOP"]
    return event_times, gti_start, gti_stop


def resolve_event_file(workdir):
    filtered_path = os.path.join(workdir, "filter_evdata.fits")
    if os.path.exists(filtered_path):
        return filtered_path

    workdir_path = Path(workdir)
    patterns = [
        "bat_downloads/*/bat/event/*bevshsl_uf*.evt*",
        "bat_downloads/*/bat/event/*bevshpo_uf*.evt*",
        "bat_downloads/*/bat/event/*.evt*",
        "bat_downloads/*_eventresult/events/*.evt*",
    ]
    for pattern in patterns:
        matches = sorted(workdir_path.glob(pattern))
        if matches:
            return str(matches[0])
    return None


def bin_light_curve(times, bin_size, gti_start, gti_stop, t0):
    time_min = t0 - 50
    time_max = t0 + 150
    bins = np.arange(time_min, time_max + bin_size, bin_size)
    bin_centers = bins[:-1] + 0.5 * bin_size
    left_edges = bin_centers - 0.5 * bin_size
    right_edges = bin_centers + 0.5 * bin_size
    in_gti = (left_edges[:, None] >= gti_start[None, :]) & (right_edges[:, None] <= gti_stop[None, :])
    good_mask = np.any(in_gti, axis=1)
    counts, _ = np.histogram(times, bins=bins)
    return bin_centers[good_mask], counts[good_mask]


def poly3_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def model(xx, t0, best_params, norm_val):
    return poly3_func(xx - t0, *best_params) * norm_val


def fit_background_linear(bin_centers, counts, t0):
    x = bin_centers
    mask = ((x >= t0 - 50) & (x < t0 - 5)) | ((x > t0 + 20) & (x <= t0 + 150))
    x_fit = x[mask]
    counts_fit = counts[mask]
    x0 = x_fit - t0
    norm_val = np.median(counts_fit)
    if norm_val == 0:
        norm_val = 1
    y = counts_fit / norm_val
    p0_ls, _ = curve_fit(poly3_func, x0, y, p0=[0, 0, 0, np.mean(y)])
    yerr = np.sqrt(np.abs(y))
    yerr[yerr == 0] = 1.0

    def log_prior(theta):
        a, b, c, d = theta
        if p0_ls[0] - 10 < a < p0_ls[0] + 10 and p0_ls[1] - 10 < b < p0_ls[1] + 10 and p0_ls[2] - 10 < c < p0_ls[2] + 10 and 0.1 * p0_ls[3] < d < 10 * p0_ls[3]:
            return 0.0
        return -np.inf

    def log_likelihood(theta, x, y, yerr):
        model_counts = np.clip(poly3_func(x, *theta), 1e-6, None) * norm_val
        y_counts = y * norm_val
        return np.sum(y_counts * np.log(model_counts) - model_counts - gammaln(y_counts + 1))

    def log_probability(theta, x, y, yerr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr)

    ndim, nwalkers = 4, 32
    p0 = p0_ls + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x0, y, yerr), threads=multiprocessing.cpu_count())
    try:
        for _ in tqdm(range(1000), desc="MCMC sampling"):
            sampler.run_mcmc(p0, 1, progress=False)
            p0 = sampler.get_last_sample().coords
    except Exception:
        sampler.run_mcmc(p0, 1000, progress=True)
    samples = sampler.get_chain(discard=200, flat=True)
    log_probs = np.array([log_probability(theta, x0, y, yerr) for theta in samples])
    best_params = samples[np.argmax(log_probs)]
    return best_params, samples, norm_val


def find_seeds(bin_centers, counts, model_bkg, counts_sub, bin_size_ms, t0, workdir, samples, norm_val):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.figure(figsize=(15, 5))
    x = bin_centers - t0
    errors = np.sqrt(counts)
    gaps = np.where(np.diff(x) > 1.1 * bin_size_ms / 1000)[0]
    segments = np.split(np.arange(len(x)), gaps + 1)
    for seg in segments:
        plt.step(x[seg], counts_sub[seg], where="mid", color="blue", alpha=0.6)
    plt.errorbar(x, counts_sub, yerr=errors, fmt="o", color="blue", alpha=0.6, label="BKG-subtracted", markersize=3, capsize=2)
    mask = (x < -5) | (x > 20)
    bkg_std = np.std(counts_sub[mask]) if np.any(mask) else np.std(counts_sub)
    with np.errstate(divide="ignore", invalid="ignore"):
        # snr = counts_sub / bkg_std
        snr = counts_sub / (model_bkg + counts)**0.5
        snr[snr < 0] = 0
    snr_max = snr.max()
    cmap = plt.get_cmap("GnBu")
    norm = plt.Normalize(0, snr_max)
    for i, xc in enumerate(x):
        color = cmap(norm(np.clip(snr[i], 0, snr_max)))
        if snr[i] > 3.5:
            plt.axvspan(xc - bin_size_ms / 2000.0, xc + bin_size_ms / 2000.0, color=color, alpha=0.3, zorder=1)
    handles, labels = plt.gca().get_legend_handles_labels()
    if "SNR > 3.5" not in labels:
        handles.append(Patch(facecolor="none", edgecolor="black", hatch="/", label="SNR > 3.5"))
    plt.legend(handles=handles, labels=[h.get_label() for h in handles])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, pad=0.01, ax=plt.gca(), alpha=0.3).set_label("SNR")
    plt.title(f"Light Curve (binning: {bin_size_ms} ms)")
    plt.xlabel("Time [s] (t - t0)")
    plt.ylabel("Counts per bin")
    plt.axhline(0, color="gray", linestyle=":", linewidth=1)
    max_snr_idx = np.where(snr == snr_max)[0][0]
    t_star = bin_centers[max_snr_idx] - t0
    window = min(45, 30 * bin_size_ms / 1000.0)
    plt.xlim(t_star - window, t_star + window)
    plt.axhspan(-3.5 * bkg_std, 3.5 * bkg_std, color="blue", alpha=0.1)
    plt.axhspan(-5 * bkg_std, 5 * bkg_std, color="blue", alpha=0.1, label="3.5σ and 5σ")
    plt.autoscale(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(workdir, f"light_curve_{bin_size_ms}ms.png"), dpi=300)

    plt.figure(figsize=(10, 5))
    for seg in segments:
        plt.step(x[seg], counts[seg], where="mid", color="black", alpha=0.6)
    n_samples = 1000
    sample_subset = samples[np.random.choice(samples.shape[0], n_samples, replace=False)] if samples.shape[0] > n_samples else samples
    model_curves = np.array([model(bin_centers, t0, params, norm_val) for params in sample_subset])
    plt.fill_between(x, np.percentile(model_curves, 5, axis=0), np.percentile(model_curves, 95, axis=0), color="red", alpha=0.2)
    plt.plot(x, model_bkg, color="red", linestyle="-", linewidth=2, label="Background model")
    plt.axvline(0, color="gray", linestyle="--", label="t0")
    plt.title(f"Raw Light Curve (binning: {bin_size_ms} ms)")
    plt.xlabel("Time [s] (t - t0)")
    plt.ylabel("Counts per bin")
    plt.legend()
    plt.grid(True)
    tmax = 150
    plt.xlim(-50, tmax)
    mask = (x >= -50) & (x <= tmax)
    ymin, ymax = counts[mask].min(), counts[mask].max()
    yrange = ymax - ymin
    plt.ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    plt.tight_layout()
    plt.savefig(os.path.join(workdir, f"light_curve_raw_{bin_size_ms}ms.png"), dpi=300)
    plt.close()
    return (bin_centers[max_snr_idx] - t0, snr_max) if snr_max > 4.0 else (None, None)


def refined_seed_search(bin_centers, counts, t_cent_init, w_init, n_trials=100):
    if len(bin_centers) == 0 or len(counts) == 0:
        return None, None
    best_sum = -np.inf
    best_t_cent = t_cent_init
    best_w = w_init
    tmin, tmax = t_cent_init - w_init / 2, t_cent_init + w_init / 2
    for _ in range(n_trials):
        w = random.uniform(0.25 * w_init, 4.0 * w_init)
        t_cent = random.uniform(tmin, tmax)
        mask = (bin_centers >= t_cent - w / 2) & (bin_centers < t_cent + w / 2)
        sum_counts = counts[mask].sum()
        if sum_counts > best_sum:
            best_sum = sum_counts
            best_t_cent = t_cent
            best_w = w
    return best_t_cent, best_w


def cust_seeds(t0, workdir):
    filename = resolve_event_file(workdir)
    if filename is None:
        logging.error("No event file found for custom seed search; skipping.")
        return None, None, None
    if not filename.endswith("filter_evdata.fits"):
        logging.info(f"Using fallback event file for custom seeds: {filename}")
    event_times, gti_start, gti_stop = load_event_times(filename)
    snr_max = 0
    seed_max = None
    dur_max = None
    seeds = []
    for bin_size_ms in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        try:
            bin_size = bin_size_ms / 1000.0
            bin_centers, counts = bin_light_curve(event_times, bin_size, gti_start, gti_stop, t0)
            if len(bin_centers) < 20:
                logging.info(f"Not enough bins for bin size {bin_size_ms} ms, skipping...")
                continue
            best_params, samples, norm_val = fit_background_linear(bin_centers, counts, t0)
            model_bkg = model(bin_centers, t0, best_params, norm_val)
            counts_sub = counts - model_bkg
            time_bin, snr_bin = find_seeds(bin_centers, counts, model_bkg, counts_sub, bin_size_ms, t0, workdir, samples, norm_val)
            if time_bin is not None or snr_bin is not None:
                if snr_bin > 4.0:
                    seeds.append([time_bin, bin_size_ms, snr_bin])
                if snr_bin > snr_max:
                    seed_max, dur_max, snr_max = time_bin, bin_size_ms, snr_bin
        except Exception:
            continue
    if seed_max is None:
        logging.info("No seeds found in the custom search")
        return None, None, None
    logging.info(f"Max SNR value of {round(snr_max, 3)} found in the time bin centered at {round(seed_max, 3)} s and duration {dur_max} ms")
    if snr_max > 5 and -20 < seed_max < 20:
        sign = "+" if seed_max >= 0 else "-"
        post_slack(os.path.join(workdir, f"light_curve_{dur_max}ms.png"), f"Trigger ID {TRIGRID}, external trigger {EXT_TRIG}:\nCustom seed search found SNR {round(snr_max, 2)}, at t0 {sign} {abs(round(seed_max, 2))} s and duration {dur_max} ms")
        post_slack(os.path.join(workdir, f"light_curve_raw_{dur_max}ms.png"), f"Trigger ID {TRIGRID}, external trigger {EXT_TRIG}:\nCustom seed search found SNR {round(snr_max, 2)}, at t0 {sign} {abs(round(seed_max, 2))} s and duration {dur_max} ms")
        if TRIG_INSTR and "IGWN" in TRIG_INSTR:
            post_telegram(os.path.join(workdir, f"light_curve_{dur_max}ms.png"), f"Trigger ID {TRIGRID}, external trigger {EXT_TRIG}:\nCustom seed search found SNR {round(snr_max, 2)}, at t0 {sign} {abs(round(seed_max, 2))} s and duration {dur_max} ms.")
            post_telegram(os.path.join(workdir, f"light_curve_raw_{dur_max}ms.png"), f"Trigger ID {TRIGRID}, external trigger {EXT_TRIG}:\nCustom seed search found SNR {round(snr_max, 2)}, at t0 {sign} {abs(round(seed_max, 2))} s and duration {dur_max} ms.")
    return seed_max, dur_max, seeds


def read_snr(workdir):
    imaging_csv = os.path.join(workdir, "imaging.csv")
    mosaic_csv = os.path.join(workdir, "mosaic.csv")
    max_snr = None
    max_snr_imaging = None
    max_snr_mosaic = None
    im_flag = False
    mos_flag = False
    if os.path.exists(imaging_csv) and os.stat(imaging_csv).st_size > 0:
        imaging_df = pd.read_csv(imaging_csv)
        if not imaging_df.empty:
            im_flag = True
            max_snr_imaging = imaging_df["SNR"].max()
            max_snr = max_snr_imaging if max_snr is None else max(max_snr, max_snr_imaging)
    if os.path.exists(mosaic_csv) and os.stat(mosaic_csv).st_size > 0:
        mosaic_df = pd.read_csv(mosaic_csv)
        if not mosaic_df.empty:
            mos_flag = True
            max_snr_mosaic = mosaic_df["SNR"].max()
            max_snr = max_snr_mosaic if max_snr is None else max(max_snr, max_snr_mosaic)
    return max_snr, im_flag, max_snr_imaging, mos_flag, max_snr_mosaic


def sort_csv(workdir, csv_file):
    df = pd.read_csv(os.path.join(workdir, csv_file))
    df.sort_values(by="SNR", ascending=False).to_csv(os.path.join(workdir, csv_file), index=False)
