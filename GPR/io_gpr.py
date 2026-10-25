"""Minimal PulseEKKO HD/DT1 import, adapted from plot_pulseekko_raw.py.

Currently supports DATA TYPE=I*2: little-endian signed 16-bit samples,
with a 128-byte header before every trace. No signal processing is applied.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class GPRData:
    amplitudes: np.ndarray       # (samples, traces), raw signed int16
    time_ns: np.ndarray          # sample times in nanoseconds
    distance_m: np.ndarray       # trace positions in metres
    header: dict
    name: str

    @property
    def dt_ns(self):
        return float(self.time_ns[1] - self.time_ns[0])


def load_header(filename):
    params = {}
    for line in Path(filename).read_text(encoding='latin-1').splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            params[key.strip().upper()] = value.strip()
    return params


def load_dt1(filename, n_traces, n_samples):
    """Return (samples, traces) of int16 values without normalization."""
    trace_header_bytes = 128
    stride = trace_header_bytes + 2 * n_samples
    raw = Path(filename).read_bytes()
    expected = n_traces * stride
    if len(raw) != expected:
        raise ValueError(f'{filename}: {len(raw)} bytes, expected {expected}.')
    data = np.empty((n_samples, n_traces), dtype=np.int16)
    for i in range(n_traces):
        data[:, i] = np.frombuffer(
            raw, dtype='<i2', count=n_samples,
            offset=i * stride + trace_header_bytes
        )
    return data


def read_pulseekko(basename):
    """Read matching basename.hd and basename.dt1; leave amplitudes untouched."""
    base = Path(basename)
    if base.suffix.lower() in ('.hd', '.dt1'):
        base = base.with_suffix('')
    header = load_header(base.with_suffix('.hd'))
    if header.get('DATA TYPE', '').upper() != 'I*2':
        raise ValueError(f"Unsupported PulseEKKO data type: {header.get('DATA TYPE')!r}")
    ntr = int(header['NUMBER OF TRACES'])
    ns = int(header['NUMBER OF PTS/TRC'])
    if ns < 2 or ntr < 1:
        raise ValueError('Invalid trace/sample count in HD header.')
    window = float(header['TOTAL TIME WINDOW'])
    start = float(header['STARTING POSITION'])
    step = float(header['STEP SIZE USED'])
    if header.get('POSITION UNITS', 'm').lower() != 'm':
        raise ValueError('This version expects position units of metres.')
    amplitudes = load_dt1(base.with_suffix('.dt1'), ntr, ns)
    # Match original plot convention: sample interval = window / number of samples.
    time_ns = np.arange(ns, dtype=float) * window / ns
    positions = start + np.arange(ntr, dtype=float) * step
    return GPRData(amplitudes, time_ns, positions, header, base.name)
