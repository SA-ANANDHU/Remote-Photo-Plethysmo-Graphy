"""
vitals/heart_rate.py
=====================
What this file does:
  - FFT-based heart rate estimation from filtered PPG signal
  - Peak detection with physiological sanity checks
  - Smoothing with exponential moving average
  - Confidence score for each estimate

Interview explanation:
  "Heart rate estimation is an FFT peak detection problem.
   The filtered signal has a dominant frequency equal to the
   heart rate in Hz. I multiply by 60 to convert to BPM.
   I also track a rolling history and apply EMA smoothing
   so the displayed number doesn't jump erratically."
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from collections import deque
from typing import Optional


HR_MIN_BPM = 42.0
HR_MAX_BPM = 240.0
HR_MIN_HZ  = HR_MIN_BPM / 60.0
HR_MAX_HZ  = HR_MAX_BPM / 60.0


class HeartRateEstimator:
    """
    Estimates heart rate from a filtered rPPG signal using FFT.

    Maintains a rolling history of estimates and applies
    Exponential Moving Average (EMA) smoothing for stable display.

    Why EMA smoothing?
      Raw FFT estimates jump between frames because:
      1. The signal window slides by 1 frame each update
      2. Small noise changes can shift the FFT peak
      EMA gives more weight to recent estimates while
      retaining memory of recent history:
        smoothed = alpha * new + (1-alpha) * previous
      alpha=0.15 means ~7 frames of effective memory.
    """

    def __init__(self, fps: float = 30.0, ema_alpha: float = 0.15):
        self.fps = fps
        self.ema_alpha = ema_alpha
        self._smoothed_hr: Optional[float] = None
        self._history = deque(maxlen=30)   # last 30 raw estimates

    def estimate(self, ppg_signal: np.ndarray) -> dict:
        """
        Estimate heart rate from filtered PPG signal.

        Args:
          ppg_signal: bandpass-filtered rPPG signal (0.7–4 Hz band)

        Returns:
          dict with:
            'hr_bpm':      smoothed heart rate (float, -1 if unreliable)
            'hr_raw':      raw FFT estimate before smoothing
            'confidence':  0.0–1.0 reliability score
            'spectrum':    (freqs, magnitudes) for plotting
        """
        if len(ppg_signal) < 30:
            return self._empty_result()

        # Detrend: remove linear drift
        signal = ppg_signal - np.mean(ppg_signal)

        # Hanning window: reduce spectral leakage
        windowed = signal * np.hanning(len(signal))

        # FFT
        n          = len(windowed)
        freqs      = fftfreq(n, d=1.0 / self.fps)
        magnitudes = np.abs(fft(windowed))

        # Keep only positive frequencies in HR range
        mask  = (freqs >= HR_MIN_HZ) & (freqs <= HR_MAX_HZ)
        if not np.any(mask):
            return self._empty_result()

        valid_freqs = freqs[mask]
        valid_mags  = magnitudes[mask]

        # Find dominant peak
        peak_idx  = np.argmax(valid_mags)
        peak_freq = valid_freqs[peak_idx]
        peak_mag  = valid_mags[peak_idx]
        hr_raw    = peak_freq * 60.0

        # Confidence: ratio of peak power to total band power
        # High confidence = one clear dominant peak
        # Low confidence  = energy spread across many frequencies
        total_power = np.sum(valid_mags) + 1e-8
        confidence  = float(peak_mag / total_power)
        confidence  = min(confidence * 3.0, 1.0)   # scale to 0–1

        # Sanity check: reject if confidence too low
        if confidence < 0.15:
            return {
                'hr_bpm': self._smoothed_hr if self._smoothed_hr else -1.0,
                'hr_raw': hr_raw,
                'confidence': confidence,
                'spectrum': (valid_freqs * 60.0, valid_mags)
            }

        # EMA smoothing
        self._history.append(hr_raw)
        if self._smoothed_hr is None:
            self._smoothed_hr = hr_raw
        else:
            self._smoothed_hr = (
                self.ema_alpha * hr_raw +
                (1 - self.ema_alpha) * self._smoothed_hr
            )

        return {
            'hr_bpm':     round(self._smoothed_hr, 1),
            'hr_raw':     round(hr_raw, 1),
            'confidence': round(confidence, 3),
            'spectrum':   (valid_freqs * 60.0, valid_mags)
        }

    def _empty_result(self) -> dict:
        return {
            'hr_bpm':     -1.0,
            'hr_raw':     -1.0,
            'confidence': 0.0,
            'spectrum':   (np.array([]), np.array([]))
        }

    @property
    def history(self) -> list:
        return list(self._history)
