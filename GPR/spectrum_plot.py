
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfreqz


def plot_filter_spectra(gpr, *, low_mhz=50, high_mhz=400,
                        order=4, max_freq_mhz=1200,
                        scale="db", figsize=(10, 5.5)):
    """
    Apply bandpass and plot spectra before/after filtering.

    Returns
    -------
    fig, ax, preview
        preview contains the filtered GPR data.
        The original gpr object is not modified.
    """
    from processing import bandpass

    # Original data
    a = np.asarray(gpr.amplitudes, dtype=float)
    dt = np.median(np.diff(gpr.time_ns))

    # Apply the existing GPR123 bandpass filter
    preview = bandpass(
        gpr, low_mhz=low_mhz,
        high_mhz=high_mhz, order=order
    )
    b = np.asarray(preview.amplitudes, dtype=float)

    # Fourier amplitude spectra (samples x traces)
    n = a.shape[0]
    freq = np.fft.rfftfreq(n, d=dt) * 1000  # MHz

    sa = np.abs(np.fft.rfft(a, axis=0)) / n
    sb = np.abs(np.fft.rfft(b, axis=0)) / n

    mean_before = sa.mean(axis=1)
    mean_after = sb.mean(axis=1)

    # Theoretical zero-phase Butterworth response
    fs = 1000 / dt
    sos = butter(
        order, [low_mhz, high_mhz],
        btype="bandpass", fs=fs, output="sos"
    )
    _, h = sosfreqz(sos, worN=freq, fs=fs)
    response = np.abs(h)**2

    # Spectral amplitude scaling
    if scale == "db":
        ref = max(mean_before.max(), np.finfo(float).tiny)

        def convert(x):
            return 20 * np.log10(
                np.maximum(x / ref, 1e-5)
            )

        traces = convert(sa)
        before = convert(mean_before)
        after = convert(mean_after)
        ylabel = "Spectral amplitude (dB)"
    elif scale == "linear":
        traces = sa
        before = mean_before
        after = mean_after
        ylabel = "Spectral amplitude"
    else:
        raise ValueError("scale must be 'db' or 'linear'")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(a.shape[1]):
        ax.plot(freq, traces[:, i],
                color="0.6", alpha=0.08, lw=0.6)

    p1, = ax.plot(freq, before, "k", lw=2,
                  label="Mean before")
    p2, = ax.plot(freq, after, "r", lw=2,
                  label="Mean after")

    ax2 = ax.twinx()
    p3, = ax2.plot(
        freq, response, "--", color="tab:blue",
        lw=2, label="Filter response"
    )

    ax.set_xlim(0, min(max_freq_mhz, fs / 2))
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

    if scale == "db":
        ax.set_ylim(-100, 5)
    else:
        ax.set_ylim(bottom=0)

    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Filter transmission")

    ax.legend(
        [p1, p2, p3],
        ["Mean before", "Mean after", "Filter response"]
    )

    fig.tight_layout()
    return fig, ax, preview

    fs = 1000.0 / dt  # MHz (time in ns)
    freq = np.fft.rfftfreq(len(ta), d=dt) * 1000.0
    spec_a = np.abs(np.fft.rfft(a, axis=0)) / len(ta)
    spec_b = np.abs(np.fft.rfft(b, axis=0)) / len(ta)
    mean_a = spec_a.mean(axis=1)
    mean_b = spec_b.mean(axis=1)

    # Exactly the theoretical Butterworth magnitude at the FFT frequencies.
    response = None
    if low_mhz is not None or high_mhz is not None:
        if low_mhz is not None and high_mhz is not None:
            cut, btype = [float(low_mhz), float(high_mhz)], 'bandpass'
        elif low_mhz is not None:
            cut, btype = float(low_mhz), 'highpass'
        else:
            cut, btype = float(high_mhz), 'lowpass'
        cuts = np.atleast_1d(cut)
        if np.any(cuts <= 0) or np.any(cuts >= fs / 2) or (len(cuts) == 2 and cuts[0] >= cuts[1]):
            raise ValueError('Cutoffs must satisfy 0 < low < high < Nyquist')
        sos = butter(order, cut, fs=fs, btype=btype, output='sos')
        _, h = sosfreqz(sos, worN=freq, fs=fs)
        response = np.abs(h) ** (2 if zero_phase else 1)

    fig, ax = plt.subplots(figsize=figsize)
    if scale == 'db':
        # Crucial: one reference for before AND after; never normalize them separately.
        ref = max(float(mean_a.max()), np.finfo(float).tiny)
        def transform(y):
            return 20 * np.log10(np.maximum(y / ref, 10 ** (db_floor / 20)))
        grey, y0, y1 = transform(spec_a), transform(mean_a), transform(mean_b)
        ax.set_ylabel('Amplitude (dB re max. mean before)')
        ax.set_ylim(db_floor, max(4, np.nanmax(y0) + 3))
    elif scale == 'linear':
        grey, y0, y1 = spec_a, mean_a, mean_b
        ax.set_ylabel('Spectral amplitude (relative)')
        ax.set_ylim(bottom=0)
    else:
        raise ValueError("scale must be 'db' or 'linear'")

    for k in range(a.shape[1]):
        ax.plot(freq, grey[:, k], color='0.55', alpha=trace_alpha, linewidth=0.5, rasterized=True)
    p_before, = ax.plot(freq, y0, color='black', lw=2.0, label='Mean before')
    p_after, = ax.plot(freq, y1, color='#c62828', lw=2.0, label='Mean after')
    axr = ax.twinx()
    labels = [p_before, p_after]
    if response is not None:
        p_resp, = axr.plot(freq, response, color='#1674bc', linestyle='--', lw=2,
                           label='Theoretical filter transmission')
        labels.append(p_resp)
    axr.set_ylim(0, 1.05)
    axr.set_ylabel('Filter transmission (0–1)', color='#1674bc')
    axr.tick_params(axis='y', colors='#1674bc')
    ax.set_xlim(0, min(fs / 2, float(max_freq_mhz)) if max_freq_mhz is not None else fs / 2)
    ax.set_xlabel('Frequency (MHz)')
    ax.grid(True, alpha=0.25)
    ax.legend(labels, [line.get_label() for line in labels], loc='upper right')
    fig.tight_layout()
    return fig, {'amplitude': ax, 'response': axr}
