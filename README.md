# BAT Glimpse

## Install

After publishing to PyPI:

```bash
python -m pip install bat-glimpse
```

From the repository root before publishing:

```bash
python -m pip install .
```

## Developer Mode

```bash
git clone <repository-url>
cd BAT-GLIMPSE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Editable mode links the installed command to this checkout, so changes under
`batglimpse/` are picked up without reinstalling the package.

Run the local checkout with:

```bash
bat-glimpse --workdir /path/to/workdir --trigtime 2026-01-01T00:00:00.000
```

or:

```bash
python -m batglimpse --workdir /path/to/workdir --trigtime 2026-01-01T00:00:00.000
```

The pipeline also expects the Swift/BAT analysis environment required by
`batanalysis` to be configured, including HEASoft/CALDB where applicable.

## Run

```bash
bat-glimpse --workdir /path/to/workdir --trigtime 2026-01-01T00:00:00.000
```

You can also run the package module directly:

```bash
python -m batglimpse --workdir /path/to/workdir --trigtime 2026-01-01T00:00:00.000
```

If `--trigtime` is omitted, BAT Glimpse reads `trigtime` from
`<workdir>/config.json`.

## Main options

- `--workdir`: required working directory for inputs and outputs.
- `--trigtime`: trigger time in `YYYY-MM-DDTHH:MM:SS.sss` format.
- `--tmin` and `--tmax`: explicit ad-hoc analysis window.
- `--pipe`: `imaging` or `mosaic` for ad-hoc analysis.
- `--ext_obsid`: override the GUANO obsid.
- `--healpix_nside`: mosaic HEALPix resolution.
- `--skyview_nprocs`: processes used while creating skyviews.
- `--mosaic_nprocs`: processes used while mosaicing.

## Outputs

BAT Glimpse writes logs, CSV detections, maps, and diagnostic plots into the
working directory. The main detection tables are `imaging.csv` and
`mosaic.csv`.

## Authentication

Set `ECHO_API_TOKEN` if Echo trigger metadata is required:

```bash
export ECHO_API_TOKEN=your-token
```
