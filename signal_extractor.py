"""
preprocessing/signal_extractor.py
===================================
What this file does:
  - Uses MediaPipe to detect face landmarks every frame
  - Extracts the forehead Region of Interest (ROI)
  - Computes mean R, G, B values from the ROI
  - Returns a clean signal buffer ready for filtering

Why this is separate from day1_rppg.py:
  day1_rppg.py was a single-file prototype.
  In a real project, signal extraction is its own module
  so you can swap the face detector or ROI method
  without touching the model or dashboard code.
  This is called Separation of Concerns — interviewers ask about this.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Optional, Tuple


# Forehead landmark indices from MediaPipe 468-point face mesh
# These are the upper-face cluster points that form the forehead region
FOREHEAD_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


class FaceROIExtractor:
    """
    Detects face landmarks and extracts mean RGB
    from the forehead region of interest every frame.

    Why forehead?
      - Highest density of superficial blood vessels
      - No facial hair (unlike cheeks/chin)
      - Minimal muscle movement artifacts
      - Flat geometry → consistent lighting
    """

    def __init__(self,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.6):

        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.prev_gray = None   # for optical flow motion detection

    def extract(self, frame: np.ndarray) -> Tuple[Optional[dict], Optional[tuple]]:
        """
        Process one frame.

        Returns:
            rgb_sample: dict with keys 'r', 'g', 'b' and 'quality' score
                        None if no face detected
            roi_box:    (x1, y1, x2, y2) for drawing on frame
                        None if no face detected
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0]
        roi = self._get_forehead_box(landmarks, h, w)

        if roi is None:
            return None, None

        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            return None, None

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None, None

        # Motion quality check using optical flow
        # High motion = low quality sample, should be weighted down
        motion_score = self._motion_score(frame)

        rgb_sample = {
            'r': float(np.mean(patch[:, :, 2])),   # OpenCV is BGR
            'g': float(np.mean(patch[:, :, 1])),
            'b': float(np.mean(patch[:, :, 0])),
            'quality': 1.0 - min(motion_score, 1.0)
        }

        return rgb_sample, roi

    def _get_forehead_box(self, landmarks, h: int, w: int) -> Optional[tuple]:
        """
        Compute bounding box of forehead landmark cluster.
        Uses upper 40% of face height as the forehead region.
        """
        xs, ys = [], []
        for idx in FOREHEAD_LANDMARKS:
            lm = landmarks.landmark[idx]
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        if not xs:
            return None

        x_min = max(0, min(xs))
        x_max = min(w, max(xs))
        y_min = max(0, min(ys))
        y_max = min(h, max(ys))

        # Keep only upper 40% of face bounding box
        face_h = y_max - y_min
        y_max_forehead = y_min + int(face_h * 0.4)

        return (x_min, y_min, x_max, y_max_forehead)

    def _motion_score(self, frame: np.ndarray) -> float:
        """
        Estimate inter-frame motion using mean optical flow magnitude.

        Why this matters:
          Head movements create illumination changes 100x larger
          than the cardiac signal. High-motion frames should be
          excluded or down-weighted before FFT.

        Returns:
          0.0 = no motion (perfect quality)
          1.0 = high motion (poor quality, discard this frame)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))   # small for speed

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.prev_gray = gray

        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return float(np.mean(magnitude)) / 10.0   # normalize to ~0-1


class SignalBuffer:
    """
    Rolling buffer that stores RGB samples over a time window.

    Why a rolling buffer?
      We always want the LATEST N seconds of signal for FFT.
      A deque with maxlen automatically discards old samples
      when new ones arrive — O(1) append, no manual slicing.
    """

    def __init__(self, fps: float = 30.0, window_seconds: float = 30.0):
        self.fps = fps
        self.window_seconds = window_seconds
        size = int(fps * window_seconds)

        self.r = deque(maxlen=size)
        self.g = deque(maxlen=size)
        self.b = deque(maxlen=size)
        self.quality = deque(maxlen=size)
        self.timestamps = deque(maxlen=size)

    def add(self, sample: dict, timestamp: float):
        """Add one RGB sample to all buffers."""
        self.r.append(sample['r'])
        self.g.append(sample['g'])
        self.b.append(sample['b'])
        self.quality.append(sample['quality'])
        self.timestamps.append(timestamp)

    def get_arrays(self) -> dict:
        """Return current buffer contents as numpy arrays."""
        return {
            'r': np.array(self.r),
            'g': np.array(self.g),
            'b': np.array(self.b),
            'quality': np.array(self.quality)
        }

    @property
    def size(self) -> int:
        return len(self.g)

    @property
    def seconds_collected(self) -> float:
        return self.size / self.fps

    def is_ready(self, min_seconds: float = 10.0) -> bool:
        """True once we have enough signal for a reliable FFT."""
        return self.seconds_collected >= min_seconds
