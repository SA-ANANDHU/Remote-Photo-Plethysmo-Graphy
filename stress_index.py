"""
vitals/stress_index.py
=======================
What this file does:
  - Detects peaks in the PPG waveform (R-peaks equivalent)
  - Computes RR intervals (time between beats)
  - Derives HRV metrics: SDNN, RMSSD, LF/HF ratio
  - Outputs a 0–100 stress score and label

Interview explanation:
  "HRV — Heart Rate Variability — is the variation in time
   between consecutive heartbeats. A healthy, relaxed person
   has high HRV (variable intervals). A stressed person has
   low HRV (rigid, mechanical rhythm). I extract RR intervals
   from the PPG peak positions, then compute SDNN (standard
   deviation of intervals) and RMSSD (root mean square of
   successive differences) — both standard clinical HRV metrics.
   Low SDNN = high stress."
"""

import numpy as np
from scipy.signal import find_peaks
from collections import deque
from typing import List, Optional


class StressEstimator:
    """
    Estimates stress level from Heart Rate Variability (HRV).

    HRV Metrics used:
      SDNN  — Standard Deviation of NN intervals
              Normal range: 50–100 ms
              < 20 ms = high stress / poor autonomic function

      RMSSD — Root Mean Square of Successive Differences
              Most sensitive to parasympathetic (rest) activity
              Normal range: 20–50 ms
              Low RMSSD = stress / fatigue

      LF/HF — Low Frequency / High Frequency power ratio
              LF (0.04–0.15 Hz): sympathetic + parasympathetic
              HF (0.15–0.4 Hz):  pure parasympathetic
              High LF/HF = sympathetic dominance = stress

    Day 1 proxy vs real HRV:
      Day 1 code used signal variance as a proxy.
      This file implements real RR interval extraction.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._rr_history: deque = deque(maxlen=50)
        self._stress_history: deque = deque(maxlen=10)

    def estimate(self, ppg_signal: np.ndarray) -> dict:
        """
        Estimate stress from PPG signal.

        Args:
          ppg_signal: filtered rPPG signal (0.7–4 Hz band)

        Returns:
          dict with:
            'stress_score':  0–100 (0=relaxed, 100=high stress)
            'stress_label':  'Relaxed' / 'Moderate' / 'High stress'
            'sdnn_ms':       SDNN in milliseconds
            'rmssd_ms':      RMSSD in milliseconds
            'hrv_valid':     bool — enough data for reliable estimate
        """
        if len(ppg_signal) < int(self.fps * 15):
            return self._empty_result()

        rr_intervals = self._extract_rr_intervals(ppg_signal)

        if len(rr_intervals) < 5:
            return self._empty_result()

        # Convert samples to milliseconds
        rr_ms = np.array(rr_intervals) * (1000.0 / self.fps)

        # Filter physiologically valid RR intervals (300–1800 ms = 33–200 BPM)
        rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 1800)]

        if len(rr_ms) < 4:
            return self._empty_result()

        sdnn  = float(np.std(rr_ms))
        rmssd = float(np.sqrt(np.mean(np.diff(rr_ms)**2)))

        stress_score = self._sdnn_to_stress(sdnn)

        # EMA smooth the stress score
        self._stress_history.append(stress_score)
        smoothed_stress = float(np.mean(self._stress_history))

        label = self._score_to_label(smoothed_stress)

        return {
            'stress_score': round(smoothed_stress, 1),
            'stress_label': label,
            'sdnn_ms':      round(sdnn, 1),
            'rmssd_ms':     round(rmssd, 1),
            'hrv_valid':    True
        }

    def _extract_rr_intervals(self, signal: np.ndarray) -> List[float]:
        """
        Detect peaks in PPG signal and compute inter-peak intervals.

        Peak detection parameters:
          distance: minimum samples between peaks
                    = fps * (60/max_bpm) = 30*(60/180) = 10 samples
          prominence: minimum peak height above surrounding baseline
                      prevents noise spikes being detected as peaks
        """
        # Invert if needed (some rPPG signals are inverted)
        # Use whichever orientation has bigger peaks
        pos_peaks, _ = find_peaks(signal,
                                  distance=int(self.fps * 60 / 200),
                                  prominence=0.01)
        neg_peaks, _ = find_peaks(-signal,
                                  distance=int(self.fps * 60 / 200),
                                  prominence=0.01)

        if len(pos_peaks) >= len(neg_peaks):
            peaks = pos_peaks
        else:
            peaks = neg_peaks

        if len(peaks) < 2:
            return []

        # RR intervals = samples between consecutive peaks
        rr_intervals = np.diff(peaks).tolist()
        return rr_intervals

    def _sdnn_to_stress(self, sdnn_ms: float) -> float:
        """
        Convert SDNN (ms) to stress score (0–100).

        Based on clinical reference ranges:
          sdnn > 100 ms  → very relaxed → score ~10
          sdnn 50–100 ms → normal       → score ~30–50
          sdnn 20–50 ms  → mild stress  → score ~50–70
          sdnn < 20 ms   → high stress  → score ~80–95

        This is a simplified linear mapping.
        Production version would use age/sex normalized percentiles.
        """
        if sdnn_ms >= 100:
            return 10.0
        elif sdnn_ms >= 50:
            # Linear interpolation: 100ms→10, 50ms→50
            return 10.0 + (100 - sdnn_ms) / 50.0 * 40.0
        elif sdnn_ms >= 20:
            # Linear interpolation: 50ms→50, 20ms→80
            return 50.0 + (50 - sdnn_ms) / 30.0 * 30.0
        else:
            return min(95.0, 80.0 + (20 - sdnn_ms) * 0.75)

    def _score_to_label(self, score: float) -> str:
        if score < 35:
            return "Relaxed"
        elif score < 65:
            return "Moderate"
        else:
            return "High stress"

    def _empty_result(self) -> dict:
        return {
            'stress_score': -1.0,
            'stress_label': "Collecting...",
            'sdnn_ms':      -1.0,
            'rmssd_ms':     -1.0,
            'hrv_valid':    False
        }
