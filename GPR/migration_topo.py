"""Topography-aware 2-D GPR diffraction-stack migration (constant velocity).

Standalone addition to the existing PulseEKKO notebook, no edits to existing
plotting.py or processing.py. Geometry: one (E,N,Z) per trace, trace order.

Physics: straight rays entirely in a constant-velocity medium, optionally
bistatic (Tx/Rx heights and along-line offsets). This is NOT a validated port
of RGPR's exact Kirchhoff implementation, nor a full true-amplitude migration.
Uses linear time interpolation and aperture-limited coherent summation.
"""
import numpy as np


def migrate_topography(gpr, geometry, *, velocity_m_ns=0.10,
                        max_depth_m=8.0, dz_m=0.05, dx_m=None,
                        aperture_m=3.0, tx_rx_sep_m=0.0,
                        f_dominant_mhz=None, fresnel_factor=2.0,
                        weighting='none', normalise='sqrt'):
    """Migrate a 2-D GPR profile onto a regular (elevation, distance) grid.

    Parameters
    ----------
    gpr : GPRData-like
        amplitudes[samples, traces], time_ns increasing; time zero corrected.
    geometry : dict
        keys distance_m and elevation_m from read_geometry_txt()/csv().
        Trace coordinates are interpreted as Tx/Rx midpoint positions.
    velocity_m_ns : float
        Constant velocity in ground (m/ns).
    max_depth_m : float
        Depth below LOWEST ground elevation that image grid will reach.
    dz_m, dx_m : float
        Output vertical and horizontal spacing; dx_m defaults to median trace step.
    aperture_m : float
        Maximum half-aperture along surface (m).
    tx_rx_sep_m : float
        Antenna separation in m; Tx and Rx at midpoint +/- sep/2.
        Heights at Tx/Rx interpolated from the measured topographic profile.
    f_dominant_mhz : float or None
        Optional approximate depth-dependent Fresnel-limited aperture using
        radius = fresnel_factor * sqrt(lambda * depth / 2),
        lambda = 1000*velocity_m_ns / f_dominant_mhz (m).
        This is a heuristic NOT the exact RGPR Fresnel weighting.
    weighting : {'none', 'geometric'}
        'geometric' multiplies by 1/sqrt(r_tx*r_rx), regularised near surface.
        This is NOT a full true-amplitude Kirchhoff obliquity/density weighting.
    normalise : {'sqrt', 'count', 'none'}
        Optional normalisation by contributing trace count.

    Returns
    -------
    distance_m, elevation_m, migrated : ndarray
        elevation_m increases upward and migrated has shape (z, x).
        Un-imaged cells above topography are NaN.
    """
    a = np.asarray(gpr.amplitudes, dtype=np.float64)
    t = np.asarray(gpr.time_ns, dtype=float)
    x = np.asarray(geometry['distance_m'], dtype=float)
    h = np.asarray(geometry['elevation_m'], dtype=float)
    if a.ndim != 2 or a.shape[1] != len(x) or len(h) != len(x) or len(t) != a.shape[0]:
        raise ValueError('Time, trace count and geometry dimensions do not match.')
    if len(x) < 2 or not np.all(np.diff(x) > 0) or not np.all(np.diff(t) > 0):
        raise ValueError('Time and along-profile distance must be increasing.')
    if not (np.isfinite(a).all() and np.isfinite(t).all() and
            np.isfinite(x).all() and np.isfinite(h).all()):
        raise ValueError('Non-finite input values.')
    vals = (velocity_m_ns, max_depth_m, dz_m, aperture_m)
    if not all(np.isfinite(v) and v > 0 for v in vals):
        raise ValueError('Velocity, max depth, dz and aperture must be positive.')
    if not np.isfinite(tx_rx_sep_m) or tx_rx_sep_m < 0:
        raise ValueError('Antenna separation must be nonnegative.')
    if f_dominant_mhz is not None and not (np.isfinite(f_dominant_mhz) and f_dominant_mhz > 0):
        raise ValueError('Dominant frequency must be positive.')
    if weighting not in ('none', 'geometric') or normalise not in ('sqrt', 'count', 'none'):
        raise ValueError('Invalid weighting or normalisation.')
    if dx_m is None:
        dx_m = float(np.median(np.diff(x)))
    if not np.isfinite(dx_m) or dx_m <= 0:
        raise ValueError('dx_m must be positive.')

    xi = np.linspace(x[0], x[-1], int(np.ceil((x[-1]-x[0])/dx_m))+1)
    zi = np.arange(np.min(h)-max_depth_m, np.max(h)+dz_m*0.5, dz_m)
    out = np.full((len(zi), len(xi)), np.nan, dtype=np.float32)
    # Tx/Rx elevations obtained from local surveyed ground at antenna coordinates.
    xtx, xrx = x-tx_rx_sep_m/2, x+tx_rx_sep_m/2
    htx = np.interp(xtx, x, h)  # edge extrapolation is constant
    hrx = np.interp(xrx, x, h)
    lambda_m = (1000.0*velocity_m_ns/f_dominant_mhz) if f_dominant_mhz else None
    tfirst, tlast = t[0], t[-1]

    for k, ximage in enumerate(xi):
        surface = float(np.interp(ximage, x, h))
        valid_z = np.flatnonzero(zi <= surface + 1e-9)
        for iz in valid_z:
            zimage = zi[iz]
            depth_local = max(surface-zimage, dz_m)
            aper = aperture_m
            if lambda_m is not None:
                fresnel = fresnel_factor*np.sqrt(lambda_m*depth_local/2.0)
                aper = min(aperture_m, max(fresnel, float(np.median(np.diff(x)))))
            near = np.flatnonzero(np.abs(x-ximage) <= aper)
            if not len(near):
                continue
            # Bistatic straight-ray travel times, no air/ground refraction.
            rt = np.hypot(ximage-xtx[near], zimage-htx[near])
            rr = np.hypot(ximage-xrx[near], zimage-hrx[near])
            tt = (rt+rr)/velocity_m_ns
            valid = (tt>=tfirst) & (tt<=tlast)
            if not np.any(valid):
                continue
            near = near[valid]
            tt = tt[valid]
            rt = rt[valid]
            rr = rr[valid]
            # Linear time interpolation; each trace is sampled separately.
            samples = np.array([np.interp(tv, t, a[:, j]) for tv, j in zip(tt, near)])
            if weighting == 'geometric':
                scale = np.maximum(np.sqrt(rt*rr), dz_m)
                samples = samples/scale
            amplitude = float(np.sum(samples))
            if normalise == 'sqrt':
                amplitude /= np.sqrt(len(samples))
            elif normalise == 'count':
                amplitude /= len(samples)
            out[iz,k] = amplitude
    return xi, zi, out


def plot_migrated_topography(distance_m, elevation_m, migrated, *,
                              ground_distance_m=None, ground_elevation_m=None,
                              title='Topography-aware diffraction-stack migration',
                              percentile=99, figsize=(12,6)):
    """Plot rising elevation upwards, independent of Matplotlib imshow origin."""
    import matplotlib.pyplot as plt
    a = np.ma.masked_invalid(np.asarray(migrated))
    finite = a.compressed()
    lim = float(np.percentile(np.abs(finite), percentile)) if len(finite) else 1.0
    lim = max(lim, 1e-12)
    fig, ax = plt.subplots(figsize=figsize)
    cm = plt.get_cmap('RdBu_r')
    im = ax.pcolormesh(distance_m, elevation_m, a, shading='auto',
                       cmap=cm, vmin=-lim, vmax=lim)
    if ground_distance_m is not None and ground_elevation_m is not None:
        ax.plot(ground_distance_m, ground_elevation_m, color='black', lw=1.5,
                label='Ground surface')
        ax.legend(loc='lower right')
    ax.set_xlabel('Distance along profile (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title(title)
    ax.set_ylim(elevation_m[0], elevation_m[-1])
    fig.colorbar(im, ax=ax, label='Migrated amplitude (relative)')
    fig.tight_layout()
    return fig, ax
