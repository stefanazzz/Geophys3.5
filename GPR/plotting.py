"""Raw PulseEKKO plotting; zero signal processing."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_radargram(gpr, percentile=99.0, figsize=(12, 6)):
    """Make original red–white–blue raw radargram; return figure and axes."""
    if not 0 < percentile <= 100:
        raise ValueError('percentile must be in (0, 100].')
    values = gpr.amplitudes
    limit = float(np.percentile(np.abs(values.astype(np.float64)), percentile))
    if limit == 0:
        limit = 1.0
    x = gpr.distance_m
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    window = float(gpr.header['TOTAL TIME WINDOW'])
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values, cmap='RdBu_r', interpolation='nearest',
                   vmin=-limit, vmax=limit, aspect='auto',
                   extent=[x[0]-dx/2, x[-1]+dx/2, window, 0])
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Two-way travel time (ns)')
    ax.set_title(f'PulseEKKO raw GPR section — {gpr.name} '
                 f'({values.shape[1]} traces, {values.shape[0]} samples/trace)')
    fig.colorbar(im, ax=ax, pad=0.02).set_label('Raw amplitude (signed integer counts)')
    fig.tight_layout()
    return fig, ax


def save_radargram(fig, basename, dpi=200):
    """Save PNG and PDF, preserving raw amplitudes in the source dataset."""
    base = Path(basename)
    png = base.with_name(base.name + '_raw_red_white_blue.png')
    pdf = base.with_name(base.name + '_raw_red_white_blue.pdf')
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    return png, pdf


def compare_radargrams(before, after, percentile=99.0, figsize=(14, 5)):
    """Compare two sections with IDENTICAL symmetric colour limits.

    Based on the percentile of the combined before/after absolute amplitudes.
    Both plots use the same physical coordinate extent; no data are scaled.
    """
    if before.amplitudes.shape != after.amplitudes.shape:
        raise ValueError('Before and after arrays must have identical shapes.')
    if not 0 < percentile <= 100:
        raise ValueError('percentile must be in (0, 100].')
    lim = float(np.percentile(np.concatenate([
        np.abs(before.amplitudes).ravel(),
        np.abs(after.amplitudes).ravel()]), percentile))
    lim = lim if lim > 0 else 1.0
    x = before.distance_m
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    extent = [x[0] - dx/2, x[-1] + dx/2, float(before.header['TOTAL TIME WINDOW']), 0]
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, arr, title in zip(axes, (before.amplitudes, after.amplitudes),
                              ('Before: raw', 'After: dewow preview')):
        im = ax.imshow(arr, cmap='RdBu_r', interpolation='nearest',
                       vmin=-lim, vmax=lim, aspect='auto', extent=extent)
        ax.set_title(title)
        ax.set_xlabel('Distance (m)')
    axes[0].set_ylabel('Two-way travel time (ns)')
    fig.colorbar(im, ax=list(axes), label='Amplitude (common colour limits)', shrink=0.85)
    return fig, axes


def plot_radargram_interactive(gpr, percentile=99.0, title=None):
    """Optional Plotly radargram with notebook/Colab zoom, pan and hover.

    Uses the same percentile contrast and time/distance coordinates as
    plot_radargram. Does not modify data. Plotly must be installed separately.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Interactive plots require Plotly: run `%pip install plotly` "
            "and restart/re-run this cell."
        ) from exc
    if not 0 < percentile <= 100:
        raise ValueError('percentile must be in (0, 100].')
    values = gpr.amplitudes
    limit = float(np.percentile(np.abs(values.astype(np.float64)), percentile))
    limit = limit if limit > 0 else 1.0
    fig = go.Figure(go.Heatmap(
        z=values, x=gpr.distance_m, y=gpr.time_ns,
        zmin=-limit, zmax=limit, zmid=0,
        colorscale='RdBu_r',
        colorbar=dict(title='Amplitude'),
        hovertemplate='Distance: %{x:.2f} m<br>Time: %{y:.2f} ns'
                      '<br>Amplitude: %{z}<extra></extra>'
    ))
    fig.update_layout(
        title=title or f'PulseEKKO GPR — {gpr.name}',
        xaxis_title='Distance (m)', yaxis_title='Two-way travel time (ns)',
        template='plotly_white', height=570,
        margin=dict(l=65, r=30, t=65, b=60),
    )
    fig.update_yaxes(autorange='reversed')
    return fig


def plot_section(gpr, title='GPR section', percentile=99, figsize=(12,6)):
    """Plot arbitrary processed GPR data with true corrected time axes."""
    a = np.asarray(gpr.amplitudes)
    lim = np.percentile(np.abs(a[np.isfinite(a)]), percentile)
    lim = max(float(lim), 1e-12)
    fig, ax = plt.subplots(figsize=figsize)
    x=gpr.distance_m
    dx=np.median(np.diff(x)) if len(x)>1 else 1.
    dt=gpr.dt_ns
    im=ax.imshow(a, extent=[x[0]-dx/2, x[-1]+dx/2,
                            gpr.time_ns[-1]+dt/2, gpr.time_ns[0]-dt/2],
                 aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim)
    ax.set(xlabel='Distance (m)', ylabel='Two-way travel time (ns)', title=title)
    fig.colorbar(im, ax=ax, label='Amplitude (display clipping only)')
    fig.tight_layout()
    return fig, ax


def compare_sections(before, after, title='Processing preview', percentile=99,
                     scaling='common'):
    """Compare radargrams using common or independent symmetric colour limits.

    scaling='common' preserves amplitude comparability across panels.
    scaling='independent' preserves each panel's contrast, ideal for gain preview.
    Both options change display only, never the GPR amplitudes.
    """
    if before.amplitudes.shape != after.amplitudes.shape:
        raise ValueError('Comparison requires arrays with matching shapes.')
    if not 0 < percentile <= 100:
        raise ValueError('percentile must be in (0, 100].')
    if scaling not in ('common', 'independent'):
        raise ValueError("scaling must be 'common' or 'independent'.")

    arrays = [np.asarray(g.amplitudes) for g in (before, after)]
    def amplitude_limit(a):
        finite = np.abs(a[np.isfinite(a)])
        return max(float(np.percentile(finite, percentile)), 1e-12) if finite.size else 1e-12

    if scaling == 'common':
        common_limit = amplitude_limit(np.concatenate([a.ravel() for a in arrays]))
        limits = [common_limit, common_limit]
    else:
        limits = [amplitude_limit(a) for a in arrays]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, dat, lim, lab in zip(axes, (before, after), limits, ('Before', 'Preview')):
        x = dat.distance_m
        dx = np.median(np.diff(x)) if len(x) > 1 else 1.
        dt = dat.dt_ns
        im = ax.imshow(dat.amplitudes, aspect='auto', cmap='RdBu_r',
                       vmin=-lim, vmax=lim,
                       extent=[x[0]-dx/2, x[-1]+dx/2,
                               dat.time_ns[-1]+dt/2, dat.time_ns[0]-dt/2])
        ax.set(title=f'{lab} (±{lim:.3g})', xlabel='Distance (m)', ylabel='Time (ns)')
        if scaling == 'independent':
            fig.colorbar(im, ax=ax, shrink=0.8, label='Amplitude (individual scale)')
    if scaling == 'common':
        fig.colorbar(im, ax=list(axes), shrink=0.8, label='Common amplitude scale')
    fig.suptitle(title)
    return fig, axes


def plot_depth_section(distance_m, z_m, amplitudes, title='Depth section',
                       percentile=99, elevation=False):
    """Plot regular distance/depth or elevation matrix with transparent NaNs."""
    a=np.asarray(amplitudes)
    lim=max(float(np.percentile(np.abs(a[np.isfinite(a)]), percentile)),1e-12)
    dx=float(np.median(np.diff(distance_m))) if len(distance_m)>1 else 1.
    dz=float(np.median(np.diff(z_m))) if len(z_m)>1 else 1.
    fig,ax=plt.subplots(figsize=(12,6))
    #cm=plt.get_cmap('RdBu_r').copy(); cm.set_bad('white')
    
    import copy
    cm = copy.copy(plt.get_cmap('RdBu_r'))
    cm.set_bad('white')
    
    im=ax.imshow(a,cmap=cm,aspect='auto',vmin=-lim,vmax=lim,
      extent=[distance_m[0]-dx/2,distance_m[-1]+dx/2,z_m[-1]+dz/2,z_m[0]-dz/2],
      origin='lower')
    if elevation:
        # elevation array increasing; top should be greatest elevation
        im.set_extent([distance_m[0]-dx/2,distance_m[-1]+dx/2,z_m[0]-dz/2,z_m[-1]+dz/2])
        ax.set_ylim(z_m[0],z_m[-1])
    
    ax.set(xlabel='Distance (m)', ylabel='Elevation (m)' if elevation else 'Depth (m)',title=title)
    fig.colorbar(im,ax=ax,label='Amplitude (display clipping only)')
    fig.tight_layout()
    return fig,ax
