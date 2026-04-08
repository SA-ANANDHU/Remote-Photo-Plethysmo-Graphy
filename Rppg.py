"""
rPPG Vitals Engine — Full Production Version
=============================================
Author : Anandhu
Project: AI-Based Contactless Vitals Estimation

What this does:
  1. Detects face using MediaPipe (468 landmarks)
  2. Extracts forehead ROI — strongest rPPG signal zone
  3. Computes mean RGB per frame → time-series signal
  4. Motion artifact rejection via optical flow
  5. CHROM algorithm — motion-robust signal extraction
  6. Bandpass filters:
       HR band  : 0.7–4.0 Hz → 42–240 BPM
       RR band  : 0.15–0.4 Hz → 9–24 breaths/min
  7. FFT peak detection → Heart Rate in BPM
  8. FFT peak detection → Respiratory Rate in br/min
  9. RR interval extraction → HRV → Stress Index (SDNN/RMSSD)
  10. EMA smoothing for stable display
  11. Live Streamlit dashboard OR OpenCV window

Run (OpenCV window):
  python rppg_full.py

Run (Streamlit dashboard):
  streamlit run rppg_full.py -- --mode dashboard

Run on video file:
  python rppg_full.py --video path/to/video.mp4

Requirements:
  pip install mediapipe opencv-python scipy numpy streamlit plotly torch
"""

# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════

import cv2
import numpy as np
from mediapipe.python.solutions import face_mesh
import time
import argparse
from collections import deque
from typing import Optional, Tuple, List

from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq


# ══════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════

# Camera
DEFAULT_FPS       = 30.0
WINDOW_SECONDS    = 30.0      # rolling buffer size
MIN_SECONDS       = 10.0      # minimum before first estimate

# Heart rate band
HR_LOW_HZ         = 0.70      # 42 BPM
HR_HIGH_HZ        = 4.00      # 240 BPM
HR_MIN_BPM        = 42.0
HR_MAX_BPM        = 240.0

# Respiratory rate band
RR_LOW_HZ         = 0.15      # 9 breaths/min
RR_HIGH_HZ        = 0.40      # 24 breaths/min
RR_MIN_BPM        = 9.0
RR_MAX_BPM        = 24.0

# Filter order (Butterworth)
FILTER_ORDER      = 4

# EMA smoothing factor (0=no update, 1=no memory)
EMA_ALPHA_HR      = 0.15
EMA_ALPHA_RR      = 0.10
EMA_ALPHA_STRESS  = 0.08

# MediaPipe forehead landmark indices
FOREHEAD_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,
    54,  103, 67,  109
]


# ══════════════════════════════════════════════════════════════
# MODULE 1 — SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════

def bandpass_filter(signal: np.ndarray,
                    low_hz: float,
                    high_hz: float,
                    fps: float,
                    order: int = FILTER_ORDER) -> np.ndarray:
    """
    Butterworth zero-phase bandpass filter.

    Why Butterworth?
      Maximally flat passband — no ripple.
      Preserves the PPG waveform shape for HRV analysis.

    Why filtfilt (zero-phase)?
      Regular filter shifts signal in time (phase delay).
      filtfilt applies forward then backward — zero phase shift.
      Critical for accurate beat timing in HRV.

    Args:
      signal  : 1D numpy array
      low_hz  : lower cutoff in Hz
      high_hz : upper cutoff in Hz
      fps     : sampling rate (camera FPS)
      order   : filter order (4 = good balance of sharpness vs stability)

    Returns:
      filtered signal, same shape as input
    """
    nyq  = fps / 2.0
    low  = np.clip(low_hz  / nyq, 0.001, 0.999)
    high = np.clip(high_hz / nyq, 0.001, 0.999)

    if low >= high or len(signal) < 3 * order + 1:
        return signal

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def normalize_channel(signal: np.ndarray) -> np.ndarray:
    """
    Divide by mean → converts absolute brightness to
    fractional change. Makes signal illumination-independent.

    Before: [142, 143, 141] ← depends on room lighting
    After:  [1.000, 1.007, 0.993] ← illumination-independent
    """
    mean = np.mean(signal)
    return signal / (mean + 1e-8)


def chrom_rppg(r: np.ndarray,
               g: np.ndarray,
               b: np.ndarray) -> np.ndarray:
    """
    CHROM algorithm — de Haan & Jeanne (2013).
    IEEE Trans. Biomedical Engineering.

    Why CHROM over raw green channel?
      Raw green works only in still, controlled conditions.
      Head movement shifts ALL channels equally (specular
      reflection change). CHROM projects RGB into chrominance
      space ORTHOGONAL to this common-mode disturbance.
      Result: motion artifacts cancel. Cardiac signal survives.

    Math:
      Rn = R/mean(R),  Gn = G/mean(G),  Bn = B/mean(B)
      X = 3Rn - 2Gn
      Y = 1.5Rn + Gn - 1.5Bn
      alpha = std(X)/std(Y)   ← cancels motion artifact
      PPG = X - alpha * Y

    Interview answer:
      "alpha scales Y to have the same variance as X so
       when subtracted, the correlated motion artifact
       cancels, leaving only the cardiac signal."
    """
    Rn = normalize_channel(r)
    Gn = normalize_channel(g)
    Bn = normalize_channel(b)

    X = 3 * Rn - 2 * Gn
    Y = 1.5 * Rn + Gn - 1.5 * Bn

    std_y = np.std(Y)
    if std_y < 1e-8:
        return X

    alpha = np.std(X) / std_y
    return X - alpha * Y


def pos_rppg(r: np.ndarray,
             g: np.ndarray,
             b: np.ndarray) -> np.ndarray:
    """
    POS algorithm — Wang et al. (2017).
    IEEE Trans. Biomedical Engineering.

    Alternative to CHROM. Often better in varying lighting.
    Know both for interviews — shows you evaluated multiple methods.

    Projects RGB onto a 2D plane of skin color variation.
    The PPG signal is orthogonal to that plane.
    """
    Cn = np.stack([
        normalize_channel(r),
        normalize_channel(g),
        normalize_channel(b)
    ], axis=0)

    # Projection matrix (Wang et al. Eq. 3)
    H = np.array([
        [0,   1, -1],
        [-2,  1,  1]
    ], dtype=float)

    S = H @ Cn   # (2, T)

    std1 = np.std(S[1])
    if std1 < 1e-8:
        return S[0]

    alpha = np.std(S[0]) / std1
    return S[0] + alpha * S[1]


def estimate_rate_fft(signal: np.ndarray,
                      fps: float,
                      min_bpm: float,
                      max_bpm: float) -> Tuple[float, float]:
    """
    FFT-based dominant frequency estimation.

    Steps:
      1. Detrend (remove linear drift)
      2. Hanning window (reduce spectral leakage)
      3. FFT magnitude spectrum
      4. Find peak in physiological frequency range
      5. Convert Hz → BPM

    Why Hanning window?
      Our 30s window has hard edges. FFT treats these
      as high-frequency discontinuities → spectral leakage.
      Hanning tapers signal to zero at both ends → clean spectrum.
      Without it: HR estimate can be 1–2 BPM off.

    Returns:
      (rate_bpm, confidence) where confidence is 0.0–1.0
    """
    n = len(signal)
    if n < 30:
        return -1.0, 0.0

    signal   = signal - np.mean(signal)
    windowed = signal * np.hanning(n)

    freqs = fftfreq(n, d=1.0 / fps)
    mags  = np.abs(fft(windowed))

    min_hz = min_bpm / 60.0
    max_hz = max_bpm / 60.0
    mask   = (freqs >= min_hz) & (freqs <= max_hz)

    if not np.any(mask):
        return -1.0, 0.0

    valid_freqs = freqs[mask]
    valid_mags  = mags[mask]

    peak_idx    = np.argmax(valid_mags)
    peak_freq   = valid_freqs[peak_idx]
    rate_bpm    = float(peak_freq * 60.0)

    # Confidence: peak power / total band power
    confidence = float(valid_mags[peak_idx] / (np.sum(valid_mags) + 1e-8))
    confidence = min(confidence * 3.0, 1.0)

    return rate_bpm, confidence


def extract_rr_intervals(ppg_signal: np.ndarray,
                          fps: float) -> np.ndarray:
    """
    Detect peaks in PPG waveform.
    Compute inter-peak intervals (RR intervals) in milliseconds.

    Why both positive and negative peaks?
      rPPG signal polarity varies between subjects and sessions.
      We try both orientations and pick the one with more peaks.

    RR intervals → HRV → stress estimation.
    """
    min_dist = int(fps * 60.0 / 200)   # min distance at 200 BPM

    pos_peaks, _ = find_peaks( ppg_signal, distance=min_dist, prominence=0.01)
    neg_peaks, _ = find_peaks(-ppg_signal, distance=min_dist, prominence=0.01)

    peaks = pos_peaks if len(pos_peaks) >= len(neg_peaks) else neg_peaks

    if len(peaks) < 3:
        return np.array([])

    rr_samples = np.diff(peaks).astype(float)
    rr_ms      = rr_samples * (1000.0 / fps)

    # Keep physiologically valid RR intervals (300–1800 ms = 33–200 BPM)
    rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 1800)]
    return rr_ms


def compute_hrv_stress(rr_ms: np.ndarray) -> Tuple[float, float, float, str]:
    """
    Compute HRV metrics and stress score from RR intervals.

    HRV Metrics:
      SDNN  — Standard Deviation of NN intervals
              Normal: 50–100 ms. Low (<20ms) = high stress.

      RMSSD — Root Mean Square of Successive Differences
              Reflects parasympathetic (rest) activity.
              Normal: 20–50 ms. Low = stress/fatigue.

    Stress score mapping:
      sdnn > 100ms → score 10  (very relaxed)
      sdnn 50–100  → score 10–50
      sdnn 20–50   → score 50–80
      sdnn < 20    → score 80–95 (high stress)

    Returns:
      (sdnn_ms, rmssd_ms, stress_score_0_100, stress_label)
    """
    if len(rr_ms) < 4:
        return -1.0, -1.0, -1.0, "Collecting..."

    sdnn  = float(np.std(rr_ms))
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms)**2))) if len(rr_ms) > 2 else -1.0

    # Map SDNN → stress score
    if sdnn >= 100:
        score = 10.0
    elif sdnn >= 50:
        score = 10.0 + (100 - sdnn) / 50.0 * 40.0
    elif sdnn >= 20:
        score = 50.0 + (50 - sdnn) / 30.0 * 30.0
    else:
        score = min(95.0, 80.0 + (20 - sdnn) * 0.75)

    label = ("Relaxed" if score < 35
             else "Moderate" if score < 65
             else "High stress")

    return sdnn, rmssd, score, label


# ══════════════════════════════════════════════════════════════
# MODULE 2 — FACE DETECTION & ROI EXTRACTION
# ══════════════════════════════════════════════════════════════

class FaceROIExtractor:
    """
    Uses MediaPipe Face Mesh to detect 468 facial landmarks.
    Extracts forehead ROI and computes mean RGB per frame.
    Includes optical flow motion scoring.

    Why forehead ROI?
      - Highest superficial blood vessel density
      - No facial hair (unlike cheeks/chin)
      - Minimal muscle movement
      - Flat geometry → consistent lighting across patch
    """

    def __init__(self):
        mp_face = face_mesh
        self.face_mesh = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.prev_gray: Optional[np.ndarray] = None

    def extract(self, frame: np.ndarray) -> Tuple[Optional[dict], Optional[tuple]]:
        """
        Process one video frame.

        Returns:
          rgb_sample : dict {r, g, b, quality} or None
          roi_box    : (x1,y1,x2,y2) or None
        """
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            self.prev_gray = None
            return None, None

        landmarks = res.multi_face_landmarks[0]
        roi = self._forehead_box(landmarks, h, w)
        if roi is None:
            return None, None

        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            return None, None

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None, None

        motion = self._motion_score(frame)

        sample = {
            'r':       float(np.mean(patch[:, :, 2])),   # BGR → R
            'g':       float(np.mean(patch[:, :, 1])),
            'b':       float(np.mean(patch[:, :, 0])),
            'quality': max(0.0, 1.0 - motion)
        }
        return sample, roi

    def _forehead_box(self, landmarks, h: int, w: int) -> Optional[tuple]:
        xs, ys = [], []
        for idx in FOREHEAD_LANDMARKS:
            lm = landmarks.landmark[idx]
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        if not xs:
            return None

        x1 = max(0, min(xs));  x2 = min(w, max(xs))
        y1 = max(0, min(ys));  y2 = min(h, max(ys))

        # Keep only upper 40% of face = forehead
        face_h = y2 - y1
        y2_fh  = y1 + int(face_h * 0.4)

        return (x1, y1, x2, y2_fh)

    def _motion_score(self, frame: np.ndarray) -> float:
        """
        Optical flow magnitude as motion quality estimate.
        High motion → low quality → down-weight this frame's sample.

        Why optical flow?
          Direct pixel difference is sensitive to lighting changes.
          Optical flow tracks actual pixel motion, separating
          movement from illumination changes.

        Returns 0.0 (no motion) to 1.0+ (high motion)
        """
        gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64)
        )

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        self.prev_gray = gray

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return float(np.mean(mag)) / 10.0


# ══════════════════════════════════════════════════════════════
# MODULE 3 — SIGNAL BUFFER
# ══════════════════════════════════════════════════════════════

class SignalBuffer:
    """
    Rolling ring buffer for RGB time-series.

    Why deque with maxlen?
      O(1) append. Automatically discards oldest samples.
      No manual slicing needed. Memory is bounded.
      At 30 FPS × 30 seconds = 900 samples per channel.
    """

    def __init__(self, fps: float = DEFAULT_FPS,
                 window_sec: float = WINDOW_SECONDS):
        self.fps = fps
        size     = int(fps * window_sec)

        self.r         = deque(maxlen=size)
        self.g         = deque(maxlen=size)
        self.b         = deque(maxlen=size)
        self.quality   = deque(maxlen=size)
        self.timestamps= deque(maxlen=size)

    def add(self, sample: dict, t: float):
        self.r.append(sample['r'])
        self.g.append(sample['g'])
        self.b.append(sample['b'])
        self.quality.append(sample['quality'])
        self.timestamps.append(t)

    def arrays(self) -> dict:
        return {
            'r':       np.array(self.r),
            'g':       np.array(self.g),
            'b':       np.array(self.b),
            'quality': np.array(self.quality)
        }

    @property
    def n(self) -> int:
        return len(self.g)

    @property
    def seconds(self) -> float:
        return self.n / self.fps

    def ready(self, min_sec: float = MIN_SECONDS) -> bool:
        return self.seconds >= min_sec


# ══════════════════════════════════════════════════════════════
# MODULE 4 — VITALS ESTIMATOR (core engine)
# ══════════════════════════════════════════════════════════════

class VitalsEstimator:
    """
    Combines all signal processing into one clean interface.
    Takes the signal buffer, outputs HR, RR, and stress.

    Uses EMA (Exponential Moving Average) smoothing on all outputs:
      smoothed = alpha * new_value + (1 - alpha) * prev_smoothed

    Why EMA?
      Raw FFT estimates jump between frames because the window
      slides by 1 frame each update. EMA provides temporal
      smoothing while staying responsive to real changes.
      alpha=0.15 means ~7 frames of effective memory.
    """

    def __init__(self, fps: float = DEFAULT_FPS):
        self.fps          = fps
        self._hr:  float  = -1.0
        self._rr:  float  = -1.0
        self._stress:float= -1.0
        self._stress_lbl  = "Collecting..."

    def update(self, buffer: SignalBuffer) -> dict:
        """
        Run full pipeline on current buffer.

        Returns dict with all vitals.
        """
        if not buffer.ready():
            pct = buffer.seconds / MIN_SECONDS * 100
            return {
                'hr': -1, 'rr': -1,
                'stress_score': -1,
                'stress_label': "Collecting...",
                'sdnn': -1, 'rmssd': -1,
                'hr_confidence': 0,
                'rr_confidence': 0,
                'progress_pct': pct
            }

        arr = buffer.arrays()
        r, g, b = arr['r'], arr['g'], arr['b']
        quality = arr['quality']

        # ── Step 1: CHROM signal extraction ──────────────────
        ppg_raw = chrom_rppg(r, g, b)

        # ── Step 2: Quality-weighted smoothing ───────────────
        # Replace low-quality (high-motion) frames with local mean
        ppg_clean = self._quality_smooth(ppg_raw, quality)

        # ── Step 3: Bandpass filter for HR and RR ────────────
        hr_signal = bandpass_filter(ppg_clean, HR_LOW_HZ, HR_HIGH_HZ, self.fps)
        rr_signal = bandpass_filter(ppg_clean, RR_LOW_HZ, RR_HIGH_HZ, self.fps)

        # ── Step 4: FFT peak detection ────────────────────────
        hr_raw, hr_conf = estimate_rate_fft(hr_signal, self.fps,
                                             HR_MIN_BPM, HR_MAX_BPM)
        rr_raw, rr_conf = estimate_rate_fft(rr_signal, self.fps,
                                             RR_MIN_BPM, RR_MAX_BPM)

        # ── Step 5: EMA smoothing ─────────────────────────────
        if hr_raw > 0 and hr_conf > 0.15:
            self._hr = (EMA_ALPHA_HR * hr_raw +
                        (1 - EMA_ALPHA_HR) * self._hr
                        if self._hr > 0 else hr_raw)

        if rr_raw > 0 and rr_conf > 0.12:
            self._rr = (EMA_ALPHA_RR * rr_raw +
                        (1 - EMA_ALPHA_RR) * self._rr
                        if self._rr > 0 else rr_raw)

        # ── Step 6: HRV stress index ──────────────────────────
        rr_intervals = extract_rr_intervals(hr_signal, self.fps)
        sdnn, rmssd, stress_raw, stress_lbl = compute_hrv_stress(rr_intervals)

        if stress_raw > 0:
            self._stress = (EMA_ALPHA_STRESS * stress_raw +
                            (1 - EMA_ALPHA_STRESS) * self._stress
                            if self._stress > 0 else stress_raw)
            self._stress_lbl = stress_lbl

        return {
            'hr':             round(self._hr,     1) if self._hr     > 0 else -1,
            'rr':             round(self._rr,     1) if self._rr     > 0 else -1,
            'stress_score':   round(self._stress, 1) if self._stress > 0 else -1,
            'stress_label':   self._stress_lbl,
            'sdnn':           round(sdnn,  1) if sdnn  > 0 else -1,
            'rmssd':          round(rmssd, 1) if rmssd > 0 else -1,
            'hr_confidence':  round(hr_conf, 3),
            'rr_confidence':  round(rr_conf, 3),
            'progress_pct':   100.0
        }

    def _quality_smooth(self, signal: np.ndarray,
                         quality: np.ndarray,
                         threshold: float = 0.3) -> np.ndarray:
        """
        Replace low-quality (high-motion) samples with local mean.
        Prevents motion spikes from corrupting FFT.
        """
        result   = signal.copy()
        low_mask = quality < threshold

        for i in np.where(low_mask)[0]:
            start = max(0, i - 3)
            end   = min(len(signal), i + 4)
            good  = signal[start:end][~low_mask[start:end]]
            if len(good) > 0:
                result[i] = np.mean(good)

        return result


# ══════════════════════════════════════════════════════════════
# MODULE 5 — OPENCV DISPLAY
# ══════════════════════════════════════════════════════════════

def draw_hud(frame: np.ndarray,
             vitals: dict,
             roi_found: bool,
             elapsed: float) -> np.ndarray:
    """
    Draw heads-up display overlay on frame.
    Semi-transparent panel + vital signs + progress bar.
    """
    h, w   = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent black panel
    cv2.rectangle(overlay, (8, 8), (330, 185), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    def txt(text, y, color=(255, 255, 255), scale=0.6, thick=1):
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    # Title
    txt("rPPG Vitals Engine", 30, (160, 160, 160), 0.5)

    # Heart Rate
    if vitals['hr'] > 0:
        c = (0, 255, 100) if 55 <= vitals['hr'] <= 100 else (0, 165, 255)
        txt(f"Heart Rate :  {vitals['hr']:.1f} BPM", 60, c, 0.65, 2)
    else:
        txt("Heart Rate :  collecting...", 60, (140, 140, 140), 0.6)

    # Respiratory Rate
    if vitals['rr'] > 0:
        txt(f"Resp Rate  :  {vitals['rr']:.1f} br/min", 90,
            (100, 210, 255), 0.6)
    else:
        txt("Resp Rate  :  collecting...", 90, (140, 140, 140), 0.6)

    # Stress
    stress_colors = {
        "Relaxed":       (0, 255, 100),
        "Moderate":      (0, 210, 255),
        "High stress":   (0, 80,  255),
        "Collecting...": (140, 140, 140)
    }
    sc = stress_colors.get(vitals['stress_label'], (200, 200, 200))
    if vitals['stress_score'] > 0:
        txt(f"Stress     :  {vitals['stress_label']} "
            f"({vitals['stress_score']:.0f}/100)", 120, sc, 0.6)
    else:
        txt(f"Stress     :  {vitals['stress_label']}", 120, sc, 0.6)

    # HRV
    if vitals['sdnn'] > 0:
        txt(f"SDNN={vitals['sdnn']:.1f}ms  "
            f"RMSSD={vitals['rmssd']:.1f}ms", 148,
            (160, 160, 160), 0.48)

    # Progress bar
    pct = min(1.0, vitals['progress_pct'] / 100.0)
    bw  = int(310 * pct)
    cv2.rectangle(frame, (16, 158), (326, 168), (50, 50, 50), -1)
    cv2.rectangle(frame, (16, 158), (16 + bw, 168), (0, 200, 120), -1)
    txt(f"Signal  {int(pct*100)}%", 182, (120, 120, 120), 0.45)

    # Face status
    fc = (0, 255, 80) if roi_found else (0, 80, 255)
    ft = "Face: detected" if roi_found else "Face: not found"
    txt(ft, h - 12, fc, 0.5)

    # Elapsed time
    cv2.putText(frame, f"{elapsed:.0f}s",
                (w - 60, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════
# MODULE 6 — MAIN LOOP (OpenCV)
# ══════════════════════════════════════════════════════════════

def run_opencv(source=0, fps: float = DEFAULT_FPS, save: str = None):
    """
    Main loop using OpenCV window.
    Works on webcam (source=0) or video file.
    Press Q to quit.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        return

    # Use detected FPS if available
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = detected_fps if detected_fps > 1 else fps
    print(f"FPS: {fps:.1f}")

    writer = None
    if save:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (fw, fh))

    extractor = FaceROIExtractor()
    buffer    = SignalBuffer(fps=fps)
    estimator = VitalsEstimator(fps=fps)

    start_t   = time.time()
    frame_idx = 0

    print("\n=== rPPG Engine — Press Q to quit ===")
    print("Keep face in frame. Sit still. Good lighting.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended.")
            break

        frame_idx += 1
        elapsed = time.time() - start_t

        # Mirror webcam
        if source == 0:
            frame = cv2.flip(frame, 1)

        # Extract ROI signal
        sample, roi_box = extractor.extract(frame)

        if sample is not None:
            buffer.add(sample, elapsed)
            if roi_box:
                x1, y1, x2, y2 = roi_box
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 100), 2)

        # Compute vitals
        vitals = estimator.update(buffer)

        # Draw HUD
        frame = draw_hud(frame, vitals, sample is not None, elapsed)

        if writer:
            writer.write(frame)

        cv2.imshow("rPPG Vitals Engine", frame)

        # Console log every 5 seconds
        if frame_idx % int(fps * 5) == 0:
            hr = f"{vitals['hr']:.1f}" if vitals['hr'] > 0 else "..."
            rr = f"{vitals['rr']:.1f}" if vitals['rr'] > 0 else "..."
            print(f"[{elapsed:.0f}s] "
                  f"HR={hr} BPM  "
                  f"RR={rr} br/min  "
                  f"Stress={vitals['stress_label']}  "
                  f"SDNN={vitals['sdnn']}ms")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final summary
    v = estimator.update(buffer)
    print("\n=== Final Vitals ===")
    print(f"Heart Rate    : {v['hr']} BPM")
    print(f"Resp Rate     : {v['rr']} br/min")
    print(f"Stress        : {v['stress_label']} ({v['stress_score']}/100)")
    print(f"SDNN          : {v['sdnn']} ms")
    print(f"RMSSD         : {v['rmssd']} ms")
    print(f"Frames        : {frame_idx}")


# ══════════════════════════════════════════════════════════════
# MODULE 7 — STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════════════

def run_streamlit():
    """
    Launch Streamlit dashboard.
    Run with: streamlit run rppg_full.py
    """
    try:
        import streamlit as st
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        print("Install streamlit and plotly: pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="rPPG Vitals Engine",
        layout="wide"
    )

    st.title("rPPG Vitals Engine")
    st.caption("Contactless vitals · No wearable · Camera only")

    # Session state
    for key, val in [
        ('running', False),
        ('hr_hist', deque(maxlen=300)),
        ('rr_hist', deque(maxlen=300)),
        ('stress_hist', deque(maxlen=300)),
        ('time_hist', deque(maxlen=300))
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

    # Sidebar
    with st.sidebar:
        st.header("Controls")
        if st.button("Start", type="primary", use_container_width=True):
            st.session_state.running = True
        if st.button("Stop", use_container_width=True):
            st.session_state.running = False

        st.divider()
        fps_set = st.slider("FPS", 15, 60, 30, step=5)
        win_set = st.slider("Window (s)", 10, 60, 30, step=5)

        st.divider()
        if st.button("Export CSV", use_container_width=True):
            if st.session_state.hr_hist:
                df  = pd.DataFrame({
                    'time_s':     list(st.session_state.time_hist),
                    'heart_rate': list(st.session_state.hr_hist),
                    'resp_rate':  list(st.session_state.rr_hist),
                    'stress':     list(st.session_state.stress_hist)
                })
                st.download_button("Download",
                                   data=df.to_csv(index=False),
                                   file_name="vitals.csv",
                                   mime="text/csv",
                                   use_container_width=True)

    # Layout
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Live feed")
        cam_ph = st.empty()
    with c2:
        st.subheader("Readings")
        m1, m2, m3 = st.columns(3)
        hr_ph  = m1.empty()
        rr_ph  = m2.empty()
        str_ph = m3.empty()
        prog_ph = st.empty()

    st.divider()
    ch1, ch2 = st.columns(2)
    with ch1:
        st.subheader("Heart rate trend")
        hr_chart = st.empty()
    with ch2:
        st.subheader("Stress trend")
        str_chart = st.empty()

    def line_chart(times, vals, label, color, ymin, ymax):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(times), y=list(vals),
            mode='lines', line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=color.replace('rgb','rgba').replace(')',',0.1)')
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=0,r=0,t=10,b=0),
            yaxis=dict(range=[ymin,ymax]),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    if st.session_state.running:
        extractor = FaceROIExtractor()
        buffer    = SignalBuffer(fps=fps_set, window_sec=win_set)
        estimator = VitalsEstimator(fps=fps_set)
        cap       = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Cannot open webcam.")
            st.session_state.running = False
            st.stop()

        start_t = time.time()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            elapsed = time.time() - start_t

            sample, roi_box = extractor.extract(frame)
            if sample:
                buffer.add(sample, elapsed)
                if roi_box:
                    x1,y1,x2,y2 = roi_box
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,100),2)

            cam_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         channels="RGB", use_column_width=True)

            vitals = estimator.update(buffer)
            pct    = vitals['progress_pct'] / 100.0
            prog_ph.progress(pct, text=f"Signal buffer {int(pct*100)}%")

            if vitals['hr'] > 0:
                hr_ph.metric("Heart Rate",
                             f"{vitals['hr']:.1f} BPM")
                rr_ph.metric("Resp Rate",
                             f"{vitals['rr']:.1f} br/min" if vitals['rr']>0 else "—")
                str_ph.metric("Stress",
                              f"{vitals['stress_score']:.0f}/100" if vitals['stress_score']>0 else "—",
                              delta=vitals['stress_label'])

                st.session_state.hr_hist.append(vitals['hr'])
                st.session_state.rr_hist.append(vitals['rr'] if vitals['rr']>0 else 0)
                st.session_state.stress_hist.append(vitals['stress_score'] if vitals['stress_score']>0 else 0)
                st.session_state.time_hist.append(elapsed)

            if len(st.session_state.hr_hist) > 2:
                t = list(st.session_state.time_hist)
                hr_chart.plotly_chart(
                    line_chart(t, list(st.session_state.hr_hist),
                               "HR", "rgb(0,200,100)", 40, 180),
                    use_container_width=True
                )
                str_chart.plotly_chart(
                    line_chart(t, list(st.session_state.stress_hist),
                               "Stress", "rgb(255,100,80)", 0, 100),
                    use_container_width=True
                )

            time.sleep(1.0 / fps_set)

        cap.release()

    else:
        cam_ph.info("Click Start in the sidebar.")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rPPG Vitals Engine")
    parser.add_argument("--video",  type=str,  default=None,
                        help="Path to video file (default: webcam)")
    parser.add_argument("--fps",    type=float, default=DEFAULT_FPS,
                        help="Camera FPS (default: 30)")
    parser.add_argument("--save",   type=str,  default=None,
                        help="Save output video to file")
    parser.add_argument("--mode",   type=str,  default="opencv",
                        choices=["opencv", "dashboard"],
                        help="opencv (default) or dashboard (Streamlit)")
    args = parser.parse_args()

    if args.mode == "dashboard":
        run_streamlit()
    else:
        source = args.video if args.video else 0
        run_opencv(source=source, fps=args.fps, save=args.save)