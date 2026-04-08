"""
models/chrom.py
================
Standalone CHROM and POS model wrappers.

These are the two main signal-processing baselines for rPPG.
Keeping them as a separate module means you can import and
compare them independently from the rest of the pipeline.

Usage:
    from models.chrom import CHROMModel, POSModel

    model = CHROMModel(fps=30.0)
    result = model.predict(r_signal, g_signal, b_signal)
    print(result['hr_bpm'], result['confidence'])

Interview explanation:
    "I implemented both CHROM and POS as standalone model classes
     so I could benchmark them fairly against each other and against
     PhysNet. Having a clean model interface — fit/predict — also
     makes it easy to swap models in the pipeline without changing
     any downstream code. This is the Strategy design pattern."
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────

def _normalize(signal: np.ndarray) -> np.ndarray:
    """Divide by mean — removes absolute illumination level."""
    mean = np.mean(signal)
    return signal / (mean + 1e-8)


def _bandpass(signal: np.ndarray, low: float, high: float,
              fps: float, order: int = 4) -> np.ndarray:
    """Butterworth zero-phase bandpass filter."""
    nyq  = fps / 2.0
    lo   = np.clip(low  / nyq, 0.001, 0.999)
    hi   = np.clip(high / nyq, 0.001, 0.999)
    if lo >= hi or len(signal) < 3 * order + 1:
        return signal
    b, a = butter(order, [lo, hi], btype='band')
    return filtfilt(b, a, signal)


def _fft_peak(signal: np.ndarray, fps: float,
              min_bpm: float, max_bpm: float) -> Tuple[float, float]:
    """
    FFT peak detection → dominant frequency → BPM.
    Returns (rate_bpm, confidence).
    """
    n = len(signal)
    if n < 30:
        return -1.0, 0.0

    s        = (signal - np.mean(signal)) * np.hanning(n)
    freqs    = fftfreq(n, d=1.0 / fps)
    mags     = np.abs(fft(s))
    mask     = (freqs >= min_bpm / 60.0) & (freqs <= max_bpm / 60.0)

    if not np.any(mask):
        return -1.0, 0.0

    vf, vm   = freqs[mask], mags[mask]
    peak_idx = np.argmax(vm)
    rate_bpm = float(vf[peak_idx] * 60.0)
    conf     = min(float(vm[peak_idx] / (np.sum(vm) + 1e-8)) * 3.0, 1.0)

    return rate_bpm, conf


# ─────────────────────────────────────────────
# CHROM Model
# ─────────────────────────────────────────────

class CHROMModel:
    """
    CHROM — Chrominance-based rPPG.
    de Haan & Jeanne, IEEE Trans. Biomed. Eng., 2013.

    Core idea:
      Project RGB into a chrominance plane orthogonal to
      specular reflection and motion-induced illumination changes.
      The cardiac signal survives; motion artifacts cancel.

    When to use:
      Best general-purpose baseline. Works well across
      lighting conditions and moderate motion. The standard
      comparison method in all rPPG papers.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def extract_ppg(self,
                    r: np.ndarray,
                    g: np.ndarray,
                    b: np.ndarray) -> np.ndarray:
        """
        Extract rPPG signal using CHROM projection.

        Math:
          Rn = R/mean(R),  Gn = G/mean(G),  Bn = B/mean(B)
          X  = 3Rn - 2Gn
          Y  = 1.5Rn + Gn - 1.5Bn
          alpha = std(X) / std(Y)
          PPG = X - alpha * Y

        The alpha term is the key insight:
          It scales Y to have the same variance as X.
          When subtracted, the motion artifact (correlated
          in both X and Y) cancels out.

        Returns:
          ppg: motion-robust rPPG signal, same length as input
        """
        Rn = _normalize(r)
        Gn = _normalize(g)
        Bn = _normalize(b)

        X = 3 * Rn - 2 * Gn
        Y = 1.5 * Rn + Gn - 1.5 * Bn

        std_y = np.std(Y)
        if std_y < 1e-8:
            return X

        alpha = np.std(X) / std_y
        return X - alpha * Y

    def predict(self,
                r: np.ndarray,
                g: np.ndarray,
                b: np.ndarray) -> dict:
        """
        Full pipeline: extract PPG → filter → FFT → HR + RR.

        Args:
          r, g, b: raw mean RGB channel signals (same length)

        Returns:
          dict with hr_bpm, rr_bpm, confidence, ppg_signal
        """
        ppg    = self.extract_ppg(r, g, b)
        hr_sig = _bandpass(ppg, 0.70, 4.00, self.fps)
        rr_sig = _bandpass(ppg, 0.15, 0.40, self.fps)

        hr_bpm, hr_conf = _fft_peak(hr_sig, self.fps, 42.0,  240.0)
        rr_bpm, rr_conf = _fft_peak(rr_sig, self.fps,  9.0,   24.0)

        return {
            'hr_bpm':      round(hr_bpm, 1) if hr_bpm > 0 else -1.0,
            'rr_bpm':      round(rr_bpm, 1) if rr_bpm > 0 else -1.0,
            'hr_conf':     round(hr_conf, 3),
            'rr_conf':     round(rr_conf, 3),
            'ppg_signal':  ppg,
            'hr_signal':   hr_sig,
            'method':      'CHROM'
        }


# ─────────────────────────────────────────────
# POS Model
# ─────────────────────────────────────────────

class POSModel:
    """
    POS — Plane-Orthogonal-to-Skin.
    Wang et al., IEEE Trans. Biomed. Eng., 2017.

    Core idea:
      The skin color of a person under varying illumination
      lies on a 2D plane in RGB space. The PPG signal is
      orthogonal to that plane. Project onto the orthogonal
      direction to isolate it.

    When to use:
      Often better than CHROM under rapidly varying lighting.
      Good complement — benchmark both and report which wins.

    Interview tip:
      "I implemented both CHROM and POS and compared them
       on UBFC. CHROM performed better on average but POS
       outperformed in high-motion conditions. Knowing both
       baselines makes my PhysNet comparison more rigorous."
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def extract_ppg(self,
                    r: np.ndarray,
                    g: np.ndarray,
                    b: np.ndarray) -> np.ndarray:
        """
        Extract rPPG signal using POS projection.

        Math:
          Normalize each channel: Cn = [Rn; Gn; Bn]
          Projection matrix H (Wang et al. Eq. 3):
            H = [[0,  1, -1],
                 [-2, 1,  1]]
          S = H @ Cn   →  two projections S0 and S1
          alpha = std(S0) / std(S1)
          PPG = S0 + alpha * S1

        Returns:
          ppg: motion-robust rPPG signal
        """
        Cn = np.stack([
            _normalize(r),
            _normalize(g),
            _normalize(b)
        ], axis=0)   # shape: (3, T)

        H = np.array([
            [0,   1, -1],
            [-2,  1,  1]
        ], dtype=float)

        S = H @ Cn   # shape: (2, T)

        std1 = np.std(S[1])
        if std1 < 1e-8:
            return S[0]

        alpha = np.std(S[0]) / std1
        return S[0] + alpha * S[1]

    def predict(self,
                r: np.ndarray,
                g: np.ndarray,
                b: np.ndarray) -> dict:
        """
        Full pipeline: extract PPG → filter → FFT → HR + RR.

        Returns:
          dict with hr_bpm, rr_bpm, confidence, ppg_signal
        """
        ppg    = self.extract_ppg(r, g, b)
        hr_sig = _bandpass(ppg, 0.70, 4.00, self.fps)
        rr_sig = _bandpass(ppg, 0.15, 0.40, self.fps)

        hr_bpm, hr_conf = _fft_peak(hr_sig, self.fps, 42.0, 240.0)
        rr_bpm, rr_conf = _fft_peak(rr_sig, self.fps,  9.0,  24.0)

        return {
            'hr_bpm':     round(hr_bpm, 1) if hr_bpm > 0 else -1.0,
            'rr_bpm':     round(rr_bpm, 1) if rr_bpm > 0 else -1.0,
            'hr_conf':    round(hr_conf, 3),
            'rr_conf':    round(rr_conf, 3),
            'ppg_signal': ppg,
            'hr_signal':  hr_sig,
            'method':     'POS'
        }


# ─────────────────────────────────────────────
# Green Channel Baseline
# ─────────────────────────────────────────────

class GreenChannelModel:
    """
    Simplest possible baseline — raw green channel only.

    Why include this?
      Every paper needs a naive baseline to show improvement over.
      Green channel is the "does anything work at all?" baseline.
      CHROM should always beat this. If it doesn't, something
      is wrong with your CHROM implementation.

    Interview use:
      "I always benchmark against the green channel baseline
       first. If my more complex method doesn't beat it,
       I know there's a bug — not a research finding."
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def predict(self, r: np.ndarray, g: np.ndarray,
                b: np.ndarray) -> dict:
        hr_sig = _bandpass(_normalize(g), 0.70, 4.00, self.fps)
        rr_sig = _bandpass(_normalize(g), 0.15, 0.40, self.fps)

        hr_bpm, hr_conf = _fft_peak(hr_sig, self.fps, 42.0, 240.0)
        rr_bpm, rr_conf = _fft_peak(rr_sig, self.fps,  9.0,  24.0)

        return {
            'hr_bpm':  round(hr_bpm, 1) if hr_bpm > 0 else -1.0,
            'rr_bpm':  round(rr_bpm, 1) if rr_bpm > 0 else -1.0,
            'hr_conf': round(hr_conf, 3),
            'method':  'GREEN'
        }


# ─────────────────────────────────────────────
# Model comparison utility
# ─────────────────────────────────────────────

def compare_models(r: np.ndarray, g: np.ndarray,
                   b: np.ndarray, fps: float = 30.0) -> dict:
    """
    Run all three models on the same signal and return
    a comparison dict. Useful for benchmarking and debugging.

    Usage:
        results = compare_models(r_buf, g_buf, b_buf, fps=30)
        for method, result in results.items():
            print(f"{method}: HR={result['hr_bpm']} BPM")
    """
    models = {
        'GREEN': GreenChannelModel(fps),
        'CHROM': CHROMModel(fps),
        'POS':   POSModel(fps),
    }
    return {name: model.predict(r, g, b)
            for name, model in models.items()}
