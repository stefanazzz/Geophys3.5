# Ground Penetrating Radar (GPR) — PulseEKKO processing

This folder is a self-contained teaching example for processing a **PulseEKKO GPR transect** using Python and Jupyter.

## Start here

Open **`GPR_Processing.ipynb`**. The notebook walks through raw-data loading, dewow, time-zero adjustment, bandpass filtering, background removal, gain, and optional topography and diffraction-stack migration. It lets the user preview processing steps before accepting them.

The companion Python modules must remain in the **same directory as the notebook**. Launch Jupyter from this folder (or ensure it is the notebook's working directory).

## Example transect included

- `line1.dt1` and `line1.hd`: raw PulseEKKO trace data and its metadata/header (keep together).
- `line1topo.txt`: example positioning/elevation data used by the topography section.
- `line1_*`: examples of intermediate or exported results (PNG, PDF, NPZ, JSON, HTML), supplied for teaching/reference. The notebook can regenerate processing outputs.

## Running locally

```bash
python -m pip install -r requirements.txt
jupyter notebook GPR_Processing.ipynb
```

When opened through a repository-root Binder session, navigate to the `GPR` directory before running the notebook. If additional dependencies are needed, merge those in `requirements.txt` into the repository-root Binder environment configuration.

## Python modules

- `io_gpr.py` — read PulseEKKO files
- `geometry.py` — positioning and topography geometry
- `processing.py` — processing operations
- `plotting.py` — plotting and comparison
- `migration_topo.py` — optional educational migration
- `export_gpr.py` — export processing results

## Teaching notes

Frequencies are in MHz, times in ns, positions/elevations in m, and velocities in m/ns. The notebook's migration is a simplified educational constant-velocity approximation; it is not a validated imaging or quantitative inversion procedure. AGC changes relative amplitudes. Read the notebook's processing notes before interpreting results.
