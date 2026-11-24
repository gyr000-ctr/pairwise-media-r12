"""Phase-vocoder pitch shifting.

The module implements the classical two-stage pitch shifter:

1. **Resample** the signal by the pitch ratio ``r = 2 ** (semitones / 12)``.
   This is what establishes the *exact* pitch ratio.
2. **Time-stretch** the signal back to its original duration with a phase
   vocoder.  Time stretching preserves pitch, so the final pitch ratio is
   exactly the resampling ratio while the output length matches the input.

The order of the two stages is chosen so the vocoder always operates on the
shorter of the two intermediate signals (resample first for upward shifts,
stretch first for downward shifts).  Since the window and hops are picked from
the rate, the number of vocoder frames stays proportional to the input length
for every shift, including the extremes of the supported range.

Frame-to-frame phase continuity uses the heterodyned phase advance with
identity phase locking (Laroche & Dolson):

    d      = wrap(phi - phi_prev - omega_expected * Ha)   # (-pi, pi]
    omega  = omega_expected + d / Ha
    psi   += omega * Hs                                   # at spectral peaks
    psi[k] = psi[peak] + phi[k] - phi[peak]               # locked region

which keeps every partial coherent across frames, so no phase discontinuity
(click) is introduced at frame boundaries and the synthesis does not suffer
from the amplitude overshoot of plain per-bin accumulation.

The signal is extended by edge replication before processing and the result is
cropped back, so the first and last samples receive complete window overlap
instead of being attenuated by the vocoder's normalisation.

The public API is small: :class:`PitchShift` with :meth:`PitchShift.shift`.
Every failure mode has its own exception type below :class:`PitchShiftError`
(itself a :class:`ValueError`), so an invalid frame size, an invalid semitone
value and non-finite sample data are all distinguishable.

NumPy is used when available purely as an accelerator; a complete pure-Python
fallback is included and the module has no mandatory third-party dependency.
Calls keep all working state local, so a single instance may be used from
several threads concurrently.
"""

from __future__ import annotations

import math
import operator

try:  # Optional accelerator only.
    import numpy as _np
except Exception:  # pragma: no cover - depends on the host environment
    _np = None

__all__ = [
    "PitchShift",
    "PitchShiftError",
    "InvalidFrameError",
    "InvalidSemitonesError",
    "InvalidInputError",
]

MAX_SEMITONES = 48.0
MIN_FRAME = 8
MAX_FRAME = 1 << 20
MIN_WINDOW = 64

_TWO_PI = 2.0 * math.pi
_PI = math.pi


# ---------------------------------------------------------------------------
# Exceptions.
# ---------------------------------------------------------------------------


class PitchShiftError(ValueError):
    """Base class for all pitch-shift argument errors."""


class InvalidFrameError(PitchShiftError):
    """The ``frame`` constructor argument is not a usable frame size."""


class InvalidSemitonesError(PitchShiftError):
    """The requested semitone shift is not a finite, in-range number."""


class InvalidInputError(PitchShiftError):
    """The sample buffer is not a finite sequence of real numbers."""


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------


def _validate_frame(frame):
    if isinstance(frame, bool):
        raise InvalidFrameError("frame must be an integer, not bool")
    if isinstance(frame, float):
        if not math.isfinite(frame) or not frame.is_integer():
            raise InvalidFrameError("frame must be an integral value")
        frame = int(frame)
    else:
        try:
            frame = operator.index(frame)
        except TypeError:
            raise InvalidFrameError("frame must be an integer") from None
    if frame < MIN_FRAME or frame > MAX_FRAME:
        raise InvalidFrameError(
            "frame size must be between %d and %d" % (MIN_FRAME, MAX_FRAME)
        )
    return int(frame)


def _validate_semitones(semitones):
    if isinstance(semitones, (bool, str, bytes, bytearray)):
        raise InvalidSemitonesError("semitones must be a real number")
    if isinstance(semitones, complex):
        raise InvalidSemitonesError("semitones must be a real number")
    if _np is not None and isinstance(semitones, _np.complexfloating):
        raise InvalidSemitonesError("semitones must be a real number")
    try:
        value = float(semitones)
    except (TypeError, ValueError):
        raise InvalidSemitonesError("semitones must be a real number") from None
    if not math.isfinite(value):
        raise InvalidSemitonesError("semitones must be finite")
    if abs(value) > MAX_SEMITONES:
        raise InvalidSemitonesError(
            "semitones out of range: |semitones| must be <= %g" % MAX_SEMITONES
        )
    return value


def _validate_samples(samples):
    """Return a new list of finite floats, or raise InvalidInputError."""
    if samples is None or isinstance(samples, (str, bytes, bytearray)):
        raise InvalidInputError("samples must be a sequence of real numbers")
    if _np is not None and isinstance(samples, _np.ndarray):
        if samples.ndim != 1:
            raise InvalidInputError("samples must be one-dimensional")
        if samples.dtype.kind in "SUOVc":
            raise InvalidInputError("samples must be real numbers")
    try:
        iterator = iter(samples)
    except TypeError:
        raise InvalidInputError(
            "samples must be an iterable of real numbers"
        ) from None

    out = []
    for item in iterator:
        if isinstance(item, (str, bytes, bytearray)):
            raise InvalidInputError("samples must contain real numbers only")
        try:
            value = float(item)
        except (TypeError, ValueError):
            raise InvalidInputError(
                "samples must contain real numbers only"
            ) from None
        if not math.isfinite(value):
            raise InvalidInputError(
                "samples must be finite (got NaN or infinity)"
            )
        out.append(value)
    return out


def _fit_length(values, length):
    """Force ``values`` to ``length`` samples, clamp to [-1, 1], return a list."""
    if _np is not None and isinstance(values, _np.ndarray):
        if len(values) < length:
            values = _np.concatenate(
                [values, _np.zeros(length - len(values), dtype=_np.float64)]
            )
        else:
            values = values[:length]
        values = _np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
        values = _np.clip(values, -1.0, 1.0)
        return values.astype(_np.float64, copy=False).tolist()

    result = []
    for value in values[:length]:
        value = float(value)
        if not math.isfinite(value):
            value = 0.0
        elif value > 1.0:
            value = 1.0
        elif value < -1.0:
            value = -1.0
        result.append(value)
    if len(result) < length:
        result.extend([0.0] * (length - len(result)))
    return result


# ---------------------------------------------------------------------------
# Geometry shared by both engines.
# ---------------------------------------------------------------------------


def _next_pow2(n):
    power = 1
    while power < n:
        power <<= 1
    return power


def _geometry(rate, frame):
    """Return ``(window, transform, analysis_hop, synthesis_hop)``.

    The time-scale factor of a phase vocoder is ``synthesis_hop / analysis_hop``.
    The *synthesis* hop is pinned near a quarter window for upward shifts (and
    the *analysis* hop for downward shifts) so the overlap add never aliases.

    Two constraints raise the window above the requested ``frame``:

    * both hops must stay at one sample or more;
    * the number of frames must stay bounded.  A quarter-window hop is
      proportional to the window, so without a floor a tiny ``frame`` would
      make the vocoder iterate once per input sample.  ``MIN_WINDOW`` caps the
      frame count at roughly ``len(samples) / 4`` in the worst case.
    """
    need = int(math.ceil(4.0 * max(rate, 1.0 / rate)))
    window = max(int(frame), need, MIN_WINDOW)
    if window % 2:
        window += 1
    transform = _next_pow2(max(window, 16))
    quarter = window // 4
    if quarter < 1:
        quarter = 1
    if rate >= 1.0:
        analysis_hop = max(1.0, quarter / rate)
        synthesis_hop = max(1, int(round(analysis_hop * rate)))
    else:
        analysis_hop = float(max(1, quarter))
        synthesis_hop = max(1, int(round(analysis_hop * rate)))
    return window, transform, analysis_hop, synthesis_hop


def _pad_amount(window, synthesis_hop):
    return int(window + synthesis_hop)


# ---------------------------------------------------------------------------
# Pure-Python engine.
# ---------------------------------------------------------------------------


class _FFTPlan(object):
    """Iterative radix-2 FFT of a fixed length, precomputed once per call."""

    __slots__ = ("n", "_rev", "_tw")

    def __init__(self, n):
        self.n = n
        rev = [0] * n
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j |= bit
            rev[i] = j
        self._rev = rev
        self._tw = [
            complex(math.cos(_TWO_PI * k / n), -math.sin(_TWO_PI * k / n))
            for k in range(n)
        ]

    def forward(self, data):
        n = self.n
        rev = self._rev
        tw = self._tw
        a = [data[rev[i]] for i in range(n)]
        size = 2
        while size <= n:
            half = size >> 1
            step = n // size
            for start in range(0, n, size):
                k = 0
                end = start + half
                for i in range(start, end):
                    t = tw[k] * a[i + half]
                    u = a[i]
                    a[i] = u + t
                    a[i + half] = u - t
                    k += step
            size <<= 1
        return a

    def inverse(self, data):
        conj = [z.conjugate() for z in data]
        transformed = self.forward(conj)
        scale = 1.0 / self.n
        return [z.conjugate() * scale for z in transformed]


class _RealFFTPlan(object):
    """Radix-2 FFT plan for a fixed real transform size.

    The real transform is computed with a half-length complex FFT over the
    even/odd interleaved signal, which halves the work compared to transforming
    the zero-padded real buffer.  ``size`` must be even (it is a power of two
    of at least 16 here).
    """

    __slots__ = ("n", "half", "_plan", "_twiddle")

    def __init__(self, n):
        if n % 2:
            raise ValueError("real FFT size must be even")
        self.n = n
        self.half = n // 2
        self._plan = _FFTPlan(n // 2)
        self._twiddle = [
            complex(math.cos(-_TWO_PI * k / n), math.sin(-_TWO_PI * k / n))
            for k in range(n // 2 + 1)
        ]

    def forward(self, data):
        """Return the ``n/2 + 1`` non-negative-frequency bins of real ``data``."""
        half = self.half
        packed = [complex(data[2 * j], data[2 * j + 1]) for j in range(half)]
        z = self._plan.forward(packed)
        tw = self._twiddle
        out = [0j] * (half + 1)
        for k in range(half + 1):
            a = z[k % half]
            b = z[(-k) % half].conjugate()
            out[k] = 0.5 * (a + b) - 0.5j * tw[k] * (a - b)
        return out

    def inverse(self, bins):
        """Inverse of :meth:`forward`; returns ``n`` real samples."""
        half = self.half
        tw = self._twiddle
        z = [0j] * half
        for k in range(half):
            a = bins[k]
            b = bins[half - k].conjugate()
            z[k] = 0.5 * (a + b) + 0.5j * tw[k].conjugate() * (a - b)
        spectrum = self._plan.inverse(z)
        out = [0.0] * self.n
        for j in range(half):
            value = spectrum[j]
            out[2 * j] = value.real
            out[2 * j + 1] = value.imag
        return out


def _hann_py(n):
    return [0.5 - 0.5 * math.cos(_TWO_PI * i / n) for i in range(n)]


def _wrap(value):
    return (value + _PI) % _TWO_PI - _PI


def _resample_py(x, ratio):
    n = len(x)
    if n == 0:
        return []
    if ratio == 1.0:
        return list(x)
    length = max(1, int(round(n / ratio)))
    if length == 1:
        return [x[0]]
    if n == 1:
        return [x[0]] * length
    out = [0.0] * length
    last = n - 1
    for j in range(length):
        pos = j * ratio
        base = int(pos)
        if base >= last:
            out[j] = x[last]
        else:
            frac = pos - base
            out[j] = x[base] * (1.0 - frac) + x[base + 1] * frac
    return out


def _spectral_peaks(mags):
    peaks = []
    for k in range(1, len(mags) - 1):
        if mags[k] > mags[k - 1] and mags[k] >= mags[k + 1] and mags[k] > 0.0:
            peaks.append(k)
    if not peaks:
        best = 0
        best_mag = -1.0
        for k, mag in enumerate(mags):
            if mag > best_mag:
                best_mag = mag
                best = k
        peaks = [best]
    # DC is a legitimate partial.  Without this the Hann window's leakage
    # lobes can make a spurious bin the lowest "peak" and DC then gets locked
    # to the wrong advance, which destroys a constant (non-zero-mean) signal.
    if len(mags) > 1 and mags[0] >= mags[1] and peaks[0] != 0:
        peaks.insert(0, 0)
    return peaks


def _stretch_py(x, rate, frame):
    n = len(x)
    if n == 0:
        return []
    if rate == 1.0:
        return list(x)

    window, transform, analysis_hop, synthesis_hop = _geometry(rate, frame)
    window_vals = _hann_py(window)
    plan = _RealFFTPlan(transform)
    nbins = transform // 2 + 1
    omega_bins = [_TWO_PI * k / transform for k in range(nbins)]

    target = max(0, int(round(n * rate)))
    out = [0.0] * (target + window + synthesis_hop)
    norm = [0.0] * len(out)
    out_len = len(out)

    sys_phase = [0.0] * nbins
    prev_phase = None
    prev_pos = 0
    m = 0

    while True:
        pos = int(round(m * analysis_hop))
        if pos >= n:
            break
        buf = [0.0] * transform
        limit = min(window, n - pos)
        for t in range(limit):
            buf[t] = x[pos + t] * window_vals[t]
        if limit < window:
            # Only reachable for the final frame; replicating the last real
            # sample keeps the frame stationary instead of embedding a step to
            # zero, which the peak tracker would otherwise turn into a click.
            last = x[n - 1] * window_vals[limit]
            for t in range(limit, window):
                buf[t] = last
        spectrum = plan.forward(buf)

        mags = [0.0] * nbins
        phases = [0.0] * nbins
        for k in range(nbins):
            z = spectrum[k]
            mags[k] = math.hypot(z.real, z.imag)
            phases[k] = math.atan2(z.imag, z.real)

        if prev_phase is None:
            sys_phase = list(phases)
        else:
            hop = pos - prev_pos
            if hop <= 0:
                hop = 1
            hop_scale = synthesis_hop / float(hop)
            peaks = _spectral_peaks(mags)
            for k in peaks:
                expected = omega_bins[k] * hop
                residual = _wrap(phases[k] - prev_phase[k] - expected)
                sys_phase[k] += (expected + residual) * hop_scale
            boundaries = [
                (peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)
            ]
            region_base = [0.0] * nbins
            peak_index = 0
            for k in range(nbins):
                while peak_index < len(boundaries) and k >= boundaries[peak_index]:
                    peak_index += 1
                peak = peaks[peak_index]
                region_base[k] = sys_phase[peak] - phases[peak]
            for k in range(nbins):
                sys_phase[k] = region_base[k] + phases[k]

        prev_phase = phases
        prev_pos = pos

        spectrum_out = [0j] * nbins
        for k in range(nbins):
            mag = mags[k]
            phase = sys_phase[k]
            spectrum_out[k] = complex(mag * math.cos(phase), mag * math.sin(phase))
        spectrum_out[0] = complex(spectrum_out[0].real, 0.0)
        spectrum_out[nbins - 1] = complex(spectrum_out[nbins - 1].real, 0.0)

        frame_out = plan.inverse(spectrum_out)
        base = m * synthesis_hop
        for t in range(window):
            p = base + t
            if p >= out_len:
                break
            ww = window_vals[t]
            out[p] += frame_out[t] * ww
            norm[p] += ww * ww
        m += 1

    result = [0.0] * target
    for i in range(target):
        nm = norm[i]
        if nm > 1e-12:
            result[i] = out[i] / nm
    return result


def _pad_py(x, pad):
    if pad <= 0 or not x:
        return list(x)
    return [x[0]] * pad + list(x) + [x[-1]] * pad


def _shift_py(data, alpha, frame):
    if alpha == 1.0:
        return list(data)
    window, _, _, synthesis_hop = _geometry(alpha, frame)
    pad = _pad_amount(window, synthesis_hop)
    padded = _pad_py(data, pad)
    if alpha >= 1.0:
        resampled = _resample_py(padded, alpha)
        processed = _stretch_py(resampled, alpha, frame)
    else:
        stretched = _stretch_py(padded, alpha, frame)
        processed = _resample_py(stretched, alpha)
    n = len(data)
    out = processed[pad:pad + n]
    if len(out) < n:
        out = out + [0.0] * (n - len(out))
    return out


# ---------------------------------------------------------------------------
# NumPy engine.
# ---------------------------------------------------------------------------


def _resample_np(x, ratio):
    n = len(x)
    if n == 0:
        return _np.zeros(0, dtype=_np.float64)
    if ratio == 1.0:
        return x.copy()
    length = max(1, int(round(n / ratio)))
    if length == 1:
        return _np.array([x[0]], dtype=_np.float64)
    if n == 1:
        return _np.full(length, x[0], dtype=_np.float64)
    positions = _np.arange(length, dtype=_np.float64) * ratio
    return _np.interp(positions, _np.arange(n, dtype=_np.float64), x)


def _stretch_np(x, rate, frame):
    n = len(x)
    if n == 0:
        return _np.zeros(0, dtype=_np.float64)
    if rate == 1.0:
        return x.copy()

    window, transform, analysis_hop, synthesis_hop = _geometry(rate, frame)
    window_vals = 0.5 - 0.5 * _np.cos(_TWO_PI * _np.arange(window) / window)
    window_norm = window_vals * window_vals
    nbins = transform // 2 + 1
    omega_bins = _TWO_PI * _np.arange(nbins, dtype=_np.float64) / transform
    bins = _np.arange(nbins)

    target = max(0, int(round(n * rate)))
    out = _np.zeros(target + window + synthesis_hop, dtype=_np.float64)
    norm = _np.zeros_like(out)
    out_len = len(out)

    sys_phase = _np.zeros(nbins, dtype=_np.float64)
    prev_phase = None
    prev_pos = 0
    m = 0

    while True:
        pos = int(round(m * analysis_hop))
        if pos >= n:
            break
        buf = _np.zeros(transform, dtype=_np.float64)
        limit = min(window, n - pos)
        buf[:limit] = x[pos:pos + limit] * window_vals[:limit]
        if limit < window:
            # Final frame only: replicate the last sample rather than stepping
            # abruptly to zero (see the pure-Python engine).
            buf[limit:window] = x[n - 1] * window_vals[limit]
        spectrum = _np.fft.rfft(buf)
        mags = _np.abs(spectrum)
        phases = _np.angle(spectrum)

        if prev_phase is None:
            sys_phase = phases.copy()
        else:
            hop = pos - prev_pos
            if hop <= 0:
                hop = 1
            interior = mags[1:-1]
            peak_mask = (interior > mags[:-2]) & (interior >= mags[2:]) & (interior > 0.0)
            peaks = _np.flatnonzero(peak_mask) + 1
            if peaks.size == 0:
                peaks = _np.array([int(_np.argmax(mags))], dtype=_np.intp)
            if mags[0] >= mags[1] and peaks[0] != 0:
                # DC is a real partial; protect it from being locked to a
                # spurious leakage-lobe peak near the bottom of the spectrum.
                peaks = _np.concatenate((_np.zeros(1, dtype=_np.intp), peaks))
            expected = omega_bins[peaks] * hop
            residual = phases[peaks] - prev_phase[peaks] - expected
            residual = (residual + _PI) % _TWO_PI - _PI
            sys_phase[peaks] += (expected + residual) * (synthesis_hop / float(hop))
            if peaks.size > 1:
                boundaries = (peaks[:-1] + peaks[1:]) // 2
                region = _np.searchsorted(boundaries, bins, side="right")
            else:
                region = _np.zeros(nbins, dtype=_np.intp)
            # Lock every bin to its region's peak phase advance.  For the
            # peaks themselves this is exactly the advanced phase.
            sys_phase = sys_phase[peaks][region] - phases[peaks][region] + phases

        prev_phase = phases
        prev_pos = pos

        reconstructed = mags * _np.exp(1j * sys_phase)
        frame_out = _np.fft.irfft(reconstructed, transform)[:window]
        base = m * synthesis_hop
        if base >= out_len:
            break
        stop = min(base + window, out_len)
        count = stop - base
        out[base:stop] += frame_out[:count] * window_vals[:count]
        norm[base:stop] += window_norm[:count]
        m += 1

    result = _np.zeros(target, dtype=_np.float64)
    usable = min(target, out_len)
    ctx = norm[:usable]
    mask = ctx > 1e-12
    result[:usable][mask] = out[:usable][mask] / ctx[mask]
    return result


def _pad_np(x, pad):
    if pad <= 0 or len(x) == 0:
        return x.copy()
    return _np.concatenate(
        [
            _np.full(pad, x[0], dtype=_np.float64),
            x,
            _np.full(pad, x[-1], dtype=_np.float64),
        ]
    )


def _shift_np(data, alpha, frame):
    x = _np.asarray(data, dtype=_np.float64)
    if alpha == 1.0:
        return x.copy()
    window, _, _, synthesis_hop = _geometry(alpha, frame)
    pad = _pad_amount(window, synthesis_hop)
    padded = _pad_np(x, pad)
    if alpha >= 1.0:
        resampled = _resample_np(padded, alpha)
        processed = _stretch_np(resampled, alpha, frame)
    else:
        stretched = _stretch_np(padded, alpha, frame)
        processed = _resample_np(stretched, alpha)
    n = len(x)
    out = processed[pad:pad + n]
    if len(out) < n:
        out = _np.concatenate([out, _np.zeros(n - len(out), dtype=_np.float64)])
    return out


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


class PitchShift:
    """Phase-vocoder pitch shifter.

    Parameters
    ----------
    frame:
        Analysis window length in samples (integer, default 1024).  Larger
        windows give finer frequency resolution, smaller windows better time
        localisation.  Must be between :data:`MIN_FRAME` and
        :data:`MAX_FRAME`.
    """

    __slots__ = ("_frame",)

    def __init__(self, frame=1024):
        self._frame = _validate_frame(frame)

    @property
    def frame(self):
        """The configured analysis frame length in samples."""
        return self._frame

    def shift(self, samples, semitones):
        """Shift ``samples`` by ``semitones`` semitones.

        Returns a new ``list`` of floats with the same length as ``samples``.
        A shift of ``0`` returns an exact copy.  The instance keeps no mutable
        state between calls, so it is safe to call concurrently from multiple
        threads.

        Raises
        ------
        InvalidFrameError
            The frame size given to the constructor is unusable.
        InvalidSemitonesError
            ``semitones`` is not a finite number in the supported range.
        InvalidInputError
            ``samples`` is not a sequence of finite real numbers.
        """
        data = _validate_samples(samples)
        semitones = _validate_semitones(semitones)

        length = len(data)
        if length == 0:
            return []
        if semitones == 0.0:
            return data

        alpha = 2.0 ** (semitones / 12.0)
        if alpha == 1.0:  # a shift so tiny the ratio underflows to unity
            return data

        if _np is not None:
            shifted = _shift_np(data, alpha, self._frame)
        else:
            shifted = _shift_py(data, alpha, self._frame)
        return _fit_length(shifted, length)
