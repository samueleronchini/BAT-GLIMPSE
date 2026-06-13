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

from . import bat_glimpse_utils as utils
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
    parser.add_argument("--trig_instr", required=False, type=str, default=None, help="Trigger instrument")
    return parser.parse_args()


def load_trigger_metadata(trigid, trig_instr):
    if trig_instr is not None:
        logging.info(f"Using provided trigger instrument: {trig_instr}")
        return [], trig_instr
    if API is None:
        logging.warning("EchoAPI is not available; continuing without external trigger metadata.")
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
    ext_trig, trig_instr = load_trigger_metadata(trigid, args.trig_instr)
    utils.set_runtime_context(workdir=workdir, trigid=trigid, ext_trig=ext_trig, trig_instr=trig_instr)
    return workdir, trigid, ext_trig, trig_instr


def create_nitrates_config(
    trigger_time,
    output_file="config.json",
    trigger_id=0,
    queue_id=0,
):

    config = {
        "ERRORS": [],
        "WARNINGS": [],
        "config": {
            "BkgPost": "true",
            "BkgPre": "true",
            "BkgSrcPosFit": "null",
            "Epeaks": [97.7, 212.1, 460.6],
            "Gammas": [0.1, 0.6, 1.1],
            "MaxDT": 20.48,
            "MaxDur": 16.384,
            "MinDT": -20.48,
            "MinDur": 0.128,
            "id": 99,
            "minSNR": 2.5,
            "name": "Default",
            "version": "0.0.0",
        },
        "queueID": queue_id,
        "triggerID": trigger_id,
        "trigtime": trigger_time,
    }

    with open(output_file, "w") as f:
        json.dump(config, f, indent=4)


def main():
    start_time = time.time()
    print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
    args = parse_args()
    workdir, trigid, ext_trig, trig_instr = prepare_runtime(args)

    create_nitrates_config(
        trigger_time=args.trigtime,
        output_file=os.path.join(args.workdir, "config.json"),
    )

    triggertime = args.trigtime
    tmin = args.tmin
    tmax = args.tmax
    ext_obsid = args.ext_obsid
    pipe = args.pipe
    healpix_nside = args.healpix_nside
    skyview_nprocs = args.skyview_nprocs
    mosaic_nprocs = args.mosaic_nprocs

    if triggertime is not None:
        logging.info("Trying using already existing data")
        with open(os.path.join(workdir, "config.json"), "r", encoding="utf-8") as handle:
            config = json.load(handle)
        triggertime = config.get("trigtime")
        fail = False
        start_time_try = time.time()
        utils.search_ext_maps(triggertime, workdir)
        guano_query(triggertime, ext_obsid, workdir, tmin, tmax, pipe, healpix_nside, skyview_nprocs, mosaic_nprocs)

    logging.info(f"Time spent: {time.time() - start_time} seconds")
    try:
        log_file = os.path.join(workdir, "batglimpse.log")
        name_id = os.path.basename(workdir)
    except Exception:
        logging.error(f"Error in copying log file: {utils.traceback.format_exc()}")
