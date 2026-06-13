import argparse
import json
import logging
import multiprocessing
import os
import time
import warnings
from pathlib import Path

from swifttools.swift_too import Clock

try:
    from EchoAPI import API
except ModuleNotFoundError:
    API = None

from . import bat_glimpse_helpers as helpers
from .bat_glimpse_pipeline import guano_query


DEFAULT_API_TOKEN = os.getenv("ECHO_API_TOKEN")


def parse_args():
    parser = argparse.ArgumentParser(description="Process some time.")
    parser.add_argument("--trigtime", required=False, type=str, help="Time in the format YYYY-MM-DDTHH:MM:SS.sss")
    parser.add_argument("--workdir", required=True, type=str, help="work directory")
    parser.add_argument("--tmin", required=False, type=str, help="min time to start")
    parser.add_argument("--tmax", required=False, type=str, help="max time to start")
    parser.add_argument("--ext_obsid", required=False, type=str, help="obsid")
    parser.add_argument("--pipe", required=False, type=str, help="pipeline, either imaging or mosaic")
    parser.add_argument("--healpix_nside", type=int, default=512, help="Nside of mosaic healpix map")
    parser.add_argument("--skyview_nprocs", type=int, default=8, help="Number of processes to use when creating skyviews in parallel")
    parser.add_argument("--mosaic_nprocs", type=int, default=8, help="Number of processes to use when creating mosaic in parallel. NOTE: ALLOCATE ~10GB OF MEMORY PER PROCESS.")
    return parser.parse_args()


def load_trigger_metadata(trigid):
    if API is None:
        logging.warning("EchoAPI is not available; continuing without external trigger metadata.")
        return [], []
    if DEFAULT_API_TOKEN is None:
        logging.warning("ECHO_API_TOKEN is not set; continuing without external trigger metadata.")
        return [], []
    api = API(api_token=DEFAULT_API_TOKEN)
    parsed_results = [json.loads(entry) for entry in api.get_trigs()]
    match_ = next((entry for entry in parsed_results if entry.get("trigid") == float(trigid)), None)
    if match_ is None:
        logging.error(f"No matching trigger found for trigid {trigid}.")
        return [], []
    return match_["event_name"], match_["trigger_instruments"]


def prepare_runtime(args):
    # Normalize path to avoid empty basename when a trailing slash is used.
    workdir = os.path.abspath(os.path.normpath(args.workdir))
    trigid = os.path.basename(workdir)[:9]
    if not trigid:
        raise ValueError(f"Unable to derive trigid from workdir: {workdir}")
    os.makedirs(workdir, exist_ok=True)
    warnings.filterwarnings("ignore")
    log_path = os.path.join(workdir, "batglimpse.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
        force=True,
    )
    ext_trig, trig_instr = load_trigger_metadata(trigid)
    helpers.set_runtime_context(workdir=workdir, trigid=trigid, ext_trig=ext_trig, trig_instr=trig_instr)
    return workdir, trigid, ext_trig, trig_instr


def main():
    start_time = time.time()
    print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
    args = parse_args()
    workdir, trigid, ext_trig, trig_instr = prepare_runtime(args)

    triggertime = args.trigtime
    tmin = args.tmin
    tmax = args.tmax
    ext_obsid = args.ext_obsid
    pipe = args.pipe
    healpix_nside = args.healpix_nside
    skyview_nprocs = args.skyview_nprocs
    mosaic_nprocs = args.mosaic_nprocs

    if triggertime is None:
        logging.info("Trying using already existing data")
        with open(os.path.join(workdir, "config.json"), "r", encoding="utf-8") as handle:
            config = json.load(handle)
        triggertime = config.get("trigtime")
        helpers.search_ext_maps(triggertime, workdir)
        fail = False
        start_time_try = time.time()
        while time.time() - start_time_try < 1800 and not fail:
            try:
                logging.info(f"Triggertime from config: {triggertime}")
                obsid = helpers.get_obsid(triggertime)
                logging.info(f"ObsID from config: {obsid}")
                triggertime_z = triggertime + ".000Z" if "." not in triggertime.split("T")[1] else triggertime + "Z"
                t0_met = Clock(utctime=triggertime_z).met
                logging.info(f"obsid: {obsid}, triggertime: {triggertime}, t0_met: {t0_met}")
                download_root = os.path.join(workdir, "bat_downloads")
                os.makedirs(download_root, exist_ok=True)
                helpers.ba.datadir(download_root)
                event = helpers.ba.BatEvent(obsid, is_guano=True)
                detmask_path = Path(f"{workdir}/detmask.fits")
                if detmask_path.exists():
                    event.detector_quality_file = detmask_path
                else:
                    logging.info("Local detmask.fits not found; using GUANO-downloaded detector quality file")

                event_path = Path(f"{workdir}/filter_evdata.fits")
                if event_path.exists():
                    event.event_files = event_path
                else:
                    logging.info("Local filter_evdata.fits not found; using GUANO-downloaded event file")

                attitude_sat_path = Path(f"{workdir}/attitude.sat")
                attitude_fits_path = Path(f"{workdir}/attitude.fits")
                if attitude_sat_path.exists():
                    event.attitude_file = attitude_sat_path
                    event.attitude = helpers.ba.Attitude.from_file(event.attitude_file)
                elif attitude_fits_path.exists():
                    event.attitude_file = attitude_fits_path
                else:
                    logging.info("No local attitude file found; using GUANO-downloaded attitude file")
                event._parse_event_file()
            except Exception as exc:
                fail = True
                logging.error(exc)
                continue
            if not fail:
                fail = guano_query(triggertime, ext_obsid, workdir, tmin, tmax, pipe, healpix_nside, skyview_nprocs, mosaic_nprocs)
            if fail:
                time.sleep(60)
            else:
                break
    else:
        helpers.search_ext_maps(triggertime, workdir)
        guano_query(triggertime, ext_obsid, workdir, tmin, tmax, pipe, healpix_nside, skyview_nprocs, mosaic_nprocs)

    logging.info(f"Time spent: {time.time() - start_time} seconds")
