"""
preprocessing/filters.py
==========================
What this file does:
  - Butterworth bandpass filter (the core noise removal step)
  - Signal normalization (removes illumination drift)
  - CHROM algorithm (de Haan & Jeanne, 2013) — better than raw green
  - Detrending and windowing helpers

Interview explanation:
  "I separate filtering into its own module because the filter
   parameters (frequency bands, order) are hyperparameters I tune
   experimentally. Keeping them isolated means I can swap a
   Butterworth for a Chebyshev or a wavelet filter without
   touching the model code."
"""

import numpy as np
from scipy.signal import butter, filtfilt, detrend


# ─────────────────────────────────────────────
# Frequency band constants
# ─────────────────────────────────────────────
HR_LOW_HZ   = 0.70   # 42 BPM  — physiological lower bound
HR_HIGH_HZ  = 4.00   # 240 BPM — physiological upper bound
RR_LOW_HZ   = 0.15   # 9 breaths/min
RR_HIGH_HZ  = 0.40   # 24 breaths/min


def bandpass_filter(signal: np.ndarray,
                    low_hz: float,
                    high_hz: float,
                    fps: float,
                    order: int = 4) -> np.ndarray:
    """
    Butterworth bandpass filter.

    Why Butterworth?
      - Maximally flat magnitude response in the passband
        (no ripple like Chebyshev) — important for preserving
        the shape of the PPG waveform for HRV analysis
      - Well-understood, widely used in biomedical signal processing
      - order=4 gives sharp cutoff without phase distortion issues

    Why filtfilt (zero-phase)?
      Regular filter introduces phase delay — the output signal
      is time-shifted relative to the input. For HR we don't care,
      but for HRV (beat timing) phase matters. filtfilt applies
      the filter forward then backward, cancelling phase shift.

    Args:
      signal:   1D numpy array, raw signal
      low_hz:   lower cutoff frequency in Hz
      high_hz:  upper cutoff frequency in Hz
      fps:      sampling rate (camera FPS)
      order:    filter order (higher = sharper cutoff, more ringing)

    Returns:
      filtered signal, same shape as input
    """
    nyq = fps / 2.0
    low  = np.clip(low_hz  / nyq, 0.001, 0.999)
    high = np.clip(high_hz / nyq, 0.001, 0.999)

    if low >= high:
        return signal

    b, a = butter(order, [low, high], btype='band')
    # filtfilt needs signal length > 3 * filter order
    if len(signal) < 3 * order + 1:
        return signal

    return filtfilt(b, a, signal)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize signal by dividing by its mean.

    Why normalize?
      Raw RGB values depend on absolute illumination — a bright room
      gives higher values than a dim room. We care about RELATIVE
      fluctuations (the heartbeat pulse), not absolute brightness.
      Dividing by mean converts absolute values to fractional changes.

      Before: [142, 143, 141, 144]   ← depends on room light
      After:  [1.000, 1.007, 0.993, 1.014]  ← illumination-independent
    """
    mean = np.mean(signal)
    if mean < 1e-8:
        return signal
    return signal / mean


def chrom_rppg(r: np.ndarray,
               g: np.ndarray,
               b: np.ndarray) -> np.ndarray:
    """
    CHROM method — de Haan & Jeanne (2013).
    "Robust Pulse Rate From Chrominance-Based rPPG"
    IEEE Transactions on Biomedical Engineering.

    Why CHROM instead of raw green channel?
      The raw green channel works in still, controlled conditions.
      But any head movement causes illumination changes that affect
      ALL three channels equally (specular reflection changes).
      CHROM projects RGB into a chrominance space that is
      mathematically ORTHOGONAL to these common-mode disturbances.
      Result: motion artifacts cancel out, cardiac signal survives.

    The math:
      Normalize each channel: Rn = R/mean(R), etc.
      X = 3Rn - 2Gn        ← chrominance projection 1
      Y = 1.5Rn + Gn - 1.5Bn ← chrominance projection 2
      alpha = std(X)/std(Y) ← balance factor
      PPG = X - alpha*Y     ← motion artifacts cancel

    Interview tip:
      "The alpha term is the key insight — it scales Y to have
       the same variance as X, so when you subtract them,
       the motion artifact (which is correlated across X and Y)
       cancels, leaving only the cardiac signal."

    Args:
      r, g, b: raw mean channel signals, same length

    Returns:
      ppg: motion-robust rPPG signal
    """
    Rn = normalize_signal(r)
    Gn = normalize_signal(g)
    Bn = normalize_signal(b)

    X = 3 * Rn - 2 * Gn
    Y = 1.5 * Rn + Gn - 1.5 * Bn

    std_x = np.std(X)
    std_y = np.std(Y)

    if std_y < 1e-8:
        return X   # fallback if Y is flat

    alpha = std_x / std_y
    ppg   = X - alpha * Y

    return ppg


def pos_rppg(r: np.ndarray,
             g: np.ndarray,
             b: np.ndarray) -> np.ndarray:
    """
    POS method — Wang et al. (2017).
    "Algorithmic Principles of Remote PPG"
    IEEE Transactions on Biomedical Engineering.

    Alternative to CHROM — often better in varying lighting.
    Good to implement both and compare in your benchmarks.
    Show interviewers you know multiple baselines.

    Projects RGB onto a plane of skin-color variation
    defined by the temporal mean of the color space.
    """
    # Temporal normalization
    Cn = np.stack([
        normalize_signal(r),
        normalize_signal(g),
        normalize_signal(b)
    ], axis=0)   # shape: (3, T)

    # POS projection matrix
    # Derived from the assumption that skin color lies on a 2D plane
    # in RGB space, with the pulse orthogonal to that plane
    H = np.array([
        [0,  1, -1],
        [-2, 1,  1]
    ], dtype=float)   # shape: (2, 3)

    S = H @ Cn   # shape: (2, T)

    # Alpha balances the two projections
    std_s0 = np.std(S[0])
    std_s1 = np.std(S[1])

    if std_s1 < 1e-8:
        return S[0]

    alpha = std_s0 / std_s1
    ppg = S[0] + alpha * S[1]

    return ppg


def apply_hanning_window(signal: np.ndarray) -> np.ndarray:
    """
    Apply Hanning window before FFT.

    Why window?
      FFT assumes the signal repeats infinitely. In reality, our
      30-second window has sharp edges at start and end. These edges
      look like high-frequency discontinuities to FFT, creating
      "spectral leakage" — energy smearing into adjacent frequencies.
      The Hanning window tapers the signal to zero at both ends,
      eliminating the edges and reducing spectral leakage.

    Effect on HR estimation:
      Without window: HR peak can be 1–2 BPM off due to leakage
      With window: cleaner spectrum, more accurate peak detection
    """
    return signal * np.hanning(len(signal))


def quality_weighted_filter(signal: np.ndarray,
                             quality: np.ndarray,
                             threshold: float = 0.3) -> np.ndarray:
    """
    Zero out low-quality samples (high motion frames) before filtering.

    Instead of completely discarding motion frames (which creates
    discontinuities), we replace them with the local mean —
    a smooth interpolation that doesn't corrupt the FFT.

    Args:
      signal:    raw PPG signal
      quality:   per-sample quality score (0=bad, 1=good)
      threshold: samples below this quality are replaced

    Returns:
      signal with low-quality frames smoothed over
    """
    result = signal.copy()
    low_q_mask = quality < threshold

    if not np.any(low_q_mask):
        return result

    # Replace low-quality samples with local mean (window of 5 samples)
    for i in np.where(low_q_mask)[0]:
        start = max(0, i - 2)
        end   = min(len(signal), i + 3)
        good_neighbors = signal[start:end][~low_q_mask[start:end]]
        if len(good_neighbors) > 0:
            result[i] = np.mean(good_neighbors)

    return result
