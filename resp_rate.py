"""
vitals/resp_rate.py
====================
What this file does:
  - Estimates respiratory rate from the low-frequency rPPG band
  - Respiratory modulation is visible in the PPG signal because
    breathing changes intrathoracic pressure, which modulates
    blood volume in facial capillaries at breathing frequency

Interview explanation:
  "Breathing modulates the rPPG signal at a much lower frequency
   than heartbeats — 0.15 to 0.4 Hz, corresponding to 9–24
   breaths per minute. I apply a separate bandpass filter to
   isolate this band, then run the same FFT peak detection.
   One signal, two vitals — no extra sensors needed."
"""

import numpy as np
from scipy.fft import fft, fftfreq
from collections import deque
from typing import Optional


RR_MIN_HZ  = 0.15    # 9 breaths/min
RR_MAX_HZ  = 0.40    # 24 breaths/min
RR_MIN_BPM = RR_MIN_HZ * 60.0
RR_MAX_BPM = RR_MAX_HZ * 60.0


class RespiratoryRateEstimator:
    """
    Estimates respiratory rate from the low-frequency component
    of the rPPG signal.

    Why this works:
      Respiratory sinus arrhythmia (RSA) — breathing modulates
      heart rate and blood pressure in a way that's visible in
      the facial blood volume pulse. The dominant low-frequency
      component of the PPG signal corresponds to breathing rate.

    Limitation:
      Needs a longer signal window than HR estimation.
      RR estimation requires at least 15–20 seconds
      to reliably resolve 0.15–0.4 Hz frequency differences.
      (Frequency resolution = 1/window_length in Hz)
    """

    def __init__(self, fps: float = 30.0, ema_alpha: float = 0.10):
        self.fps = fps
        self.ema_alpha = ema_alpha
        self._smoothed_rr: Optional[float] = None
        self._history = deque(maxlen=20)

    def estimate(self, rr_signal: np.ndarray) -> dict:
        """
        Estimate respiratory rate from 0.15–0.4 Hz filtered signal.

        Args:
          rr_signal: bandpass-filtered signal (0.15–0.4 Hz band)
                     Needs at least 15 seconds (450 samples at 30fps)

        Returns:
          dict with 'rr_bpm', 'rr_raw', 'confidence'
        """
        # Need longer window for reliable low-frequency estimation
        min_samples = int(self.fps * 15)
        if len(rr_signal) < min_samples:
            return self._empty_result()

        signal   = rr_signal - np.mean(rr_signal)
        windowed = signal * np.hanning(len(signal))

        n          = len(windowed)
        freqs      = fftfreq(n, d=1.0 / self.fps)
        magnitudes = np.abs(fft(windowed))

        mask = (freqs >= RR_MIN_HZ) & (freqs <= RR_MAX_HZ)
        if not np.any(mask):
            return self._empty_result()

        valid_freqs = freqs[mask]
        valid_mags  = magnitudes[mask]

        peak_idx    = np.argmax(valid_mags)
        peak_freq   = valid_freqs[peak_idx]
        peak_mag    = valid_mags[peak_idx]
        rr_raw      = peak_freq * 60.0

        # Confidence score
        total_power = np.sum(valid_mags) + 1e-8
        confidence  = min(float(peak_mag / total_power) * 3.0, 1.0)

        if confidence < 0.12:
            return {
                'rr_bpm':     self._smoothed_rr if self._smoothed_rr else -1.0,
                'rr_raw':     rr_raw,
                'confidence': confidence
            }

        self._history.append(rr_raw)
        if self._smoothed_rr is None:
            self._smoothed_rr = rr_raw
        else:
            self._smoothed_rr = (
                self.ema_alpha * rr_raw +
                (1 - self.ema_alpha) * self._smoothed_rr
            )

        return {
            'rr_bpm':     round(self._smoothed_rr, 1),
            'rr_raw':     round(rr_raw, 1),
            'confidence': round(confidence, 3)
        }

    def _empty_result(self) -> dict:
        return {'rr_bpm': -1.0, 'rr_raw': -1.0, 'confidence': 0.0}
