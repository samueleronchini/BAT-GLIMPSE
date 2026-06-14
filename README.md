# BAT-GLIMPSE

## Requirements

HEASoft needs to be installed. We recommend to install it via Conda as described [here](https://www.anaconda.com/products/distribution), doing
```bash
conda create -n henv heasoft \
  -c https://heasarc.gsfc.nasa.gov/FTP/software/conda/ \
  -c conda-forge

```
In the same environment, install BAT-GLIMPSE and NITRATES as described in the next two steps. CALDB environment variables need to be defined as described [here](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_remote_access.html)

## Install

The package can be installed via pip, with Python 3.10 or greater.

```bash
pip install bat-glimpse
```

## Optional: installing NITRATES

In order to have the data setup managed by NITRATES, we need to install it locally using:

```bash
git clone git@github.com:Swift-BAT/NITRATES.git
cd NITRATES
python -m pip install -e .
```

## Developer Mode

```bash
git clone git@github.com:samueleronchini/BAT-GLIMPSE.git
cd BAT-GLIMPSE
python -m pip install -e .
```


## Run commands

```bash
bat-glimpse --workdir </path/to/workdir> --trigtime <trigtime>
```

You can also run the package module directly:

```bash
python -m batglimpse --workdir </path/to/workdir> --trigtime <trigtime>
```

## Main options

- `--workdir`: required working directory for inputs and outputs.
- `--trigtime`: trigger time in `YYYY-MM-DDTHH:MM:SS.sss` format.
- `--tmin` and `--tmax`: explicit ad-hoc analysis window.
- `--pipe`: `imaging` or `mosaic` for ad-hoc analysis.
- `--ext_obsid`: override the GUANO obsid.
- `--healpix_nside`: mosaic HEALPix resolution.
- `--skyview_nprocs`: processes used while creating skyviews.
- `--mosaic_nprocs`: processes used while mosaicing.
- `--trig_instr`: Name of the triggering instrument

For `--trig_instr` use `IGWN` when it's a GW. This allows to create the preliminary maps with the partial coding distribution. Otherwise,by default the code searches for a Fermi-GBM map.

## External Map Search

### Fermi localization

- Uses `gdt.missions.fermi` if installed.
- Computes a Fermi trigger ID from `trigtime`.
- Downloads the localization and renames it to `ext_loc_fermi_glg_*.fits`.

### GW localization (IGWN only)

- Queries GraceDb for the most relevant FITS sky map.
- Downloads it to `workdir/ext_loc_*.fits`.

## Branch A: Ad-hoc analysis (`tmin/tmax/pipe` provided)

- If `pipe == imaging`:
  - Runs `imaging()` once for `[tmin, tmax]`.
- If `pipe == mosaic`:
  - Runs `mosaic()` once for `[tmin, tmax]`.
- If max SNR >= 6:
  - Sorts CSVs and posts results to Slack/Telegram.

### Branch B: Default analysis (no explicit window)

1. **NITRATES time seeds**:
   - If `time_seeds.csv` exists and is non-empty:
     - Sort by `snr` and take top 10 seeds.
     - Call `imaging()` with those time windows.
   - If empty or missing, log and continue.
2. **SNR check**:
   - Reads SNR values from CSVs.
   - Posts to Slack/Telegram if SNR >= 6.
3. **Custom seed search**:
   - Runs `cust_seeds()`; if seeds are found, refine and re-image/mosaic:
     - If max SNR already >= 20, skip refinement.
     - For each seed within +/- 20 s:
       - Refine the seed center and duration.
       - If duration <= 0.2 s: run `imaging()`.
       - If 0.2 s <= duration < 15 s:
         - Run `mosaic()` if interval intersects a slew interval.
         - Otherwise run `imaging()`.

## Imaging Algorithm Details

### `imaging(t0, event, workdir, ...)`

- Energy range: 15-350 keV.
- Creates skyview with:
  - `aperture=CALDB:DETECTION`
  - `pcodethresh=0.01`
- Source detection parameters:
  - `snrthresh=5.5`, `srcdetect=yes`
- Filters detections:
  - `NAME` contains `UNKNOWN`.
  - `SNR > 5` and `CENT_SNR > 5`.
- Writes `imaging.csv` with RA, Dec, SNR, CENT_SNR, partial coding, detect status, dt/duration, and energy bounds.

## Mosaic Algorithm Details

### `mosaic(t0, event, workdir, ...)`

- Energy range: 15-350 keV.
- Initial duration `dt_0 = tmax - tmin`.
- Builds time bins in 0.2 s steps; uses a 3-bin fallback for short windows.
- Creates skyviews in parallel, then filters to those with:
  - `sky_img`, `pcode_img`, and `bkg_stddev_img` present.
- If 0 valid skyviews: double `dt_0` and retry.
- If 1 valid skyview: fall back to that skyview.
- Otherwise mosaic with `ba.parallel.mosaic_skyview()`.
- Detects sources with `snrthresh=5.5`.
- Accepts sources with `psffwhm_separation > 1`.
- Writes `mosaic.csv` with RA, Dec, SNR, t_start, t_end, and energy bounds.

# Run examples


Always pass trigger time explicitly.

```bash
python run_bat_glimpse.py \
  --workdir /absolute/path/to/workdir \
  --trigtime 2020-03-25T03:18:35.000
```

Optional ad-hoc window:

```bash
python run_bat_glimpse.py \
  --workdir /absolute/path/to/workdir \
  --trigtime 2020-03-25T03:18:35.000 \
  --pipe imaging \
  --tmin -2.0 \
  --tmax 6.0
```

Mosaic run with explicit process counts:

```bash
python run_bat_glimpse.py \
  --workdir /absolute/path/to/workdir \
  --trigtime 2020-03-25T03:18:35.000 \
  --pipe mosaic \
  --tmin -5.0 \
  --tmax 15.0 \
  --healpix_nside 512 \
  --skyview_nprocs 8 \
  --mosaic_nprocs 8
```
