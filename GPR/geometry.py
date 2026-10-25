"""Optional trace-level positioning/elevation CSV loading and elevation gridding."""
import csv
import numpy as np


def read_geometry_csv(path, gpr):
    """Require one row per trace, ordered trace indices 1..N, columns x,y,elevation.

    x and y are survey coordinates in metres; return cumulative distance and elevation.
    Also accept existing distance_m in preference to x,y when no XY available.
    """
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    ntr = gpr.amplitudes.shape[1]
    if len(rows) != ntr:
        raise ValueError(f'Geometry has {len(rows)} rows; section has {ntr} traces.')
    if 'trace' not in rows[0] or 'elevation' not in rows[0]:
        raise ValueError('Geometry CSV requires trace and elevation columns.')
    idx = np.array([int(r['trace']) for r in rows])
    if not np.array_equal(idx, np.arange(1, ntr+1)):
        raise ValueError('trace column must be ordered 1,2,...,N.')
    elevation = np.array([float(r['elevation']) for r in rows])
    if {'x','y'}.issubset(rows[0]):
        xx = np.array([float(r['x']) for r in rows]); yy = np.array([float(r['y']) for r in rows])
        distance = np.r_[0., np.cumsum(np.hypot(np.diff(xx),np.diff(yy)))]
    elif 'distance_m' in rows[0]:
        distance = np.array([float(r['distance_m']) for r in rows])
    else:
        distance = gpr.distance_m.copy()
    if not (np.all(np.isfinite(elevation)) and np.all(np.isfinite(distance))
            and np.all(np.diff(distance)>0)):
        raise ValueError('Invalid or non-increasing geometry coordinates.')
    return {'distance_m': distance, 'elevation_m': elevation,
            'x': xx if 'xx' in locals() else None,
            'y': yy if 'yy' in locals() else None}


def elevation_grid(gpr, geometry, velocity_m_ns=0.10, dz_m=0.05):
    """Grid traces into an elevation section for plotting (not migration).

    z=elevation - v*max(time,0)/2. Uses linear interpolation in each column;
    outside measured range is NaN. Use after time-zero correction.
    """
    if velocity_m_ns<=0 or dz_m<=0:
        raise ValueError('velocity and dz must be positive.')
    elev = geometry['elevation_m']
    depth = np.maximum(gpr.time_ns, 0)*velocity_m_ns/2
    zmax = float(np.max(elev)); zmin=float(np.min(elev)-np.max(depth))
    zz = np.arange(zmin, zmax+dz_m, dz_m)
    out = np.full((len(zz),len(elev)), np.nan, dtype=np.float32)
    for j, h in enumerate(elev):
        ztrace = h-depth
        # Remove negative-time duplicate depths
        valid = np.r_[True, np.diff(ztrace) < -1e-12]
        out[:,j] = np.interp(zz, ztrace[valid][::-1],
                              gpr.amplitudes[valid,j][::-1], left=np.nan, right=np.nan)
    return zz, out


def read_geometry_txt(path, gpr):
    """Read whitespace-delimited E N Z coordinates, one row per GPR trace.

    Rows must follow acquisition order. Coordinates and elevations are metres.
    """
    values = np.loadtxt(path, skiprows=1, ndmin=2)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError('Expected three numeric columns: E N Z.')
    ntr = gpr.amplitudes.shape[1]
    if len(values) != ntr:
        raise ValueError(f'Topography has {len(values)} rows; GPR has {ntr} traces.')
    if not np.isfinite(values).all():
        raise ValueError('Topography contains invalid coordinates/elevations.')
    x, y, elevation = values.T
    distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    if not np.all(np.diff(distance) > 0):
        raise ValueError('Trace positions must be distinct and ordered.')
    return {'distance_m': distance, 'elevation_m': elevation, 'x': x, 'y': y}
