"""
dashboard_app.py
=================
Standalone Streamlit dashboard for rPPG Vitals Engine.

Run with:
  streamlit run dashboard_app.py

No argparse. No __main__ block. Streamlit runs this top-to-bottom.
"""

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from collections import deque
from typing import Optional, Tuple

from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import mediapipe as mp


# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
DEFAULT_FPS    = 30.0
WINDOW_SECONDS = 30.0
MIN_SECONDS    = 10.0
HR_LOW_HZ      = 0.70
HR_HIGH_HZ     = 4.00
RR_LOW_HZ      = 0.15
RR_HIGH_HZ     = 0.40
EMA_HR         = 0.15
EMA_RR         = 0.10
EMA_STRESS     = 0.08

FOREHEAD_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,
    54,  103, 67,  109
]


# ══════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════

def bandpass_filter(signal, low_hz, high_hz, fps, order=4):
    nyq  = fps / 2.0
    low  = np.clip(low_hz  / nyq, 0.001, 0.999)
    high = np.clip(high_hz / nyq, 0.001, 0.999)
    if low >= high or len(signal) < 3 * order + 1:
        return signal
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def chrom_rppg(r, g, b):
    def norm(x): return x / (np.mean(x) + 1e-8)
    Rn, Gn, Bn = norm(r), norm(g), norm(b)
    X = 3*Rn - 2*Gn
    Y = 1.5*Rn + Gn - 1.5*Bn
    std_y = np.std(Y)
    if std_y < 1e-8: return X
    return X - (np.std(X) / std_y) * Y


def estimate_hr_fft(signal, fps, min_bpm, max_bpm):
    n = len(signal)
    if n < 30: return -1.0, 0.0
    s = (signal - np.mean(signal)) * np.hanning(n)
    freqs = fftfreq(n, d=1.0/fps)
    mags  = np.abs(fft(s))
    mask  = (freqs >= min_bpm/60) & (freqs <= max_bpm/60)
    if not np.any(mask): return -1.0, 0.0
    vf, vm = freqs[mask], mags[mask]
    pi   = np.argmax(vm)
    conf = min(float(vm[pi] / (np.sum(vm) + 1e-8)) * 3.0, 1.0)
    return float(vf[pi] * 60.0), conf


def compute_stress(signal, fps):
    min_dist = int(fps * 60 / 200)
    pp, _ = find_peaks( signal, distance=min_dist, prominence=0.01)
    np_, _ = find_peaks(-signal, distance=min_dist, prominence=0.01)
    peaks = pp if len(pp) >= len(np_) else np_
    if len(peaks) < 3: return -1.0, -1.0, -1.0, "Collecting..."
    rr = np.diff(peaks) * (1000.0 / fps)
    rr = rr[(rr >= 300) & (rr <= 1800)]
    if len(rr) < 3: return -1.0, -1.0, -1.0, "Collecting..."
    sdnn  = float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr)**2))) if len(rr)>2 else -1.0
    if   sdnn >= 100: score = 10.0
    elif sdnn >= 50:  score = 10.0 + (100-sdnn)/50*40
    elif sdnn >= 20:  score = 50.0 + (50-sdnn)/30*30
    else:             score = min(95.0, 80.0+(20-sdnn)*0.75)
    label = "Relaxed" if score<35 else "Moderate" if score<65 else "High stress"
    return sdnn, rmssd, score, label


# ══════════════════════════════════════════════════════════════
# FACE EXTRACTOR
# ══════════════════════════════════════════════════════════════

class FaceExtractor:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.prev_gray = None

    def extract(self, frame):
        h, w = frame.shape[:2]
        res  = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None, None

        lm  = res.multi_face_landmarks[0]
        xs  = [int(lm.landmark[i].x * w) for i in FOREHEAD_LANDMARKS]
        ys  = [int(lm.landmark[i].y * h) for i in FOREHEAD_LANDMARKS]

        x1,x2 = max(0,min(xs)), min(w,max(xs))
        y1,y2 = max(0,min(ys)), min(h,max(ys))
        y2    = y1 + int((y2-y1)*0.4)

        if x2<=x1 or y2<=y1: return None, None

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0: return None, None

        # Motion score
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(64,64))
        motion = 0.0
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
            motion = float(np.mean(np.sqrt(flow[...,0]**2+flow[...,1]**2)))/10
        self.prev_gray = gray

        return {
            'r': float(np.mean(patch[:,:,2])),
            'g': float(np.mean(patch[:,:,1])),
            'b': float(np.mean(patch[:,:,0])),
            'quality': max(0.0, 1.0-motion)
        }, (x1,y1,x2,y2)


# ══════════════════════════════════════════════════════════════
# PAGE SETUP
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Remote PhotoPlethysmoGraphy Vitals Engine",
    page_icon="❤",
    layout="wide"
)

st.title("Remote PhotoPlethysmoGraphy Vitals Engine")
st.caption("Contactless heart rate · respiratory rate · stress index — camera only, no wearable")

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════

defaults = {
    'running':      False,
    'hr_hist':      deque(maxlen=300),
    'rr_hist':      deque(maxlen=300),
    'stress_hist':  deque(maxlen=300),
    'time_hist':    deque(maxlen=300),
    'hr_ema':       -1.0,
    'rr_ema':       -1.0,
    'stress_ema':   -1.0,
    'stress_label': "Collecting...",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Vitals Control Panel")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", type="primary", use_container_width=True):
            st.session_state.running = True
            st.session_state.hr_ema  = -1.0
            st.session_state.rr_ema  = -1.0
    with col2:
        if st.button("Stop", use_container_width=True):
            st.session_state.running = False

    st.divider()
    st.subheader("Settings")
    fps_set = st.slider("Camera FPS",       15, 60, 30, step=5)
    win_set = st.slider("Signal window (s)", 10, 60, 30, step=5)

    st.divider()
    st.subheader("Session export")
    if len(st.session_state.hr_hist) > 0:
        df  = pd.DataFrame({
            'time_s':     list(st.session_state.time_hist),
            'heart_rate': list(st.session_state.hr_hist),
            'resp_rate':  list(st.session_state.rr_hist),
            'stress':     list(st.session_state.stress_hist),
        })
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False),
            file_name="vitals_session.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()
    st.caption("Anandhu — Remote PhotoPlethysmoGraphy Vitals Engine v1.0")
    st.caption("· AI/ML Engineer")

# ══════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ════════════════════════════════════════════════
left, right = st.columns([1, 1])

with left:
    st.subheader("Live camera feed")
    cam_ph   = st.empty()
    prog_ph  = st.empty()

with right:
    st.subheader("Current vitals")
    m1, m2, m3 = st.columns(3)
    hr_ph  = m1.empty()
    rr_ph  = m2.empty()
    str_ph = m3.empty()

    st.markdown("---")
    hrv_ph = st.empty()

st.markdown("---")
chart_l, chart_r = st.columns(2)
with chart_l:
    st.subheader("Heart rate trend")
    hr_chart_ph = st.empty()
with chart_r:
    st.subheader("Stress trend")
    stress_chart_ph = st.empty()


# ══════════════════════════════════════════════════════════════
# CHART HELPER
# ══════════════════════════════════════════════════════════════

def make_chart(times, values, color, ymin, ymax, ylabel):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(times), y=list(values),
        mode='lines',
        line=dict(color=color, width=2.5),
        fill='tozeroy',
        fillcolor=color.replace('rgb','rgba').replace(')',',0.08)')
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(range=[ymin, ymax], title=ylabel,
                   gridcolor='rgba(128,128,128,0.15)'),
        xaxis=dict(title="Time (s)",
                   gridcolor='rgba(128,128,128,0.15)'),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# ══════════════════════════════════════════════════════════════
# NOT RUNNING STATE
# ══════════════════════════════════════════════════════════════

if not st.session_state.running:
    cam_ph.info("Click **Start** in the sidebar to begin monitoring.")
    hr_ph.metric("Heart Rate",  "— BPM")
    rr_ph.metric("Resp Rate",   "— br/min")
    str_ph.metric("Stress",     "—")
    st.stop()


# ══════════════════════════════════════════════════════════════
# MONITORING LOOP
# ══════════════════════════════════════════════════════════════

extractor = FaceExtractor()

buf_size = int(fps_set * win_set)
r_buf  = deque(maxlen=buf_size)
g_buf  = deque(maxlen=buf_size)
b_buf  = deque(maxlen=buf_size)
q_buf  = deque(maxlen=buf_size)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Cannot open webcam. Check camera permissions.")
    st.session_state.running = False
    st.stop()

start_t    = time.time()
min_buf    = int(fps_set * MIN_SECONDS)

while st.session_state.running:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera read failed.")
        break

    frame   = cv2.flip(frame, 1)
    elapsed = time.time() - start_t

    # Extract signal
    sample, roi = extractor.extract(frame)

    if sample is not None:
        r_buf.append(sample['r'])
        g_buf.append(sample['g'])
        b_buf.append(sample['b'])
        q_buf.append(sample['quality'])
        if roi:
            x1,y1,x2,y2 = roi
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,100),2)
            # Label on frame
            cv2.putText(frame, "Forehead ROI",
                        (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0,255,100), 1, cv2.LINE_AA)

    # Show camera
    cam_ph.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_column_width=True
    )

    # Progress bar
    n_samples = len(g_buf)
    pct = min(1.0, n_samples / buf_size)
    prog_ph.progress(pct,
        text=f"Signal buffer: {n_samples}/{buf_size} samples "
             f"({n_samples/fps_set:.0f}s / {win_set}s)")

    # Compute vitals when enough data
    if n_samples >= min_buf:
        r = np.array(r_buf)
        g = np.array(g_buf)
        b = np.array(b_buf)
        q = np.array(q_buf)

        # Quality smoothing — replace motion frames with local mean
        ppg = chrom_rppg(r, g, b)
        low_q = q < 0.3
        for i in np.where(low_q)[0]:
            s = max(0,i-3); e = min(len(ppg),i+4)
            good = ppg[s:e][~low_q[s:e]]
            if len(good) > 0: ppg[i] = np.mean(good)

        # Filter
        hr_sig = bandpass_filter(ppg, HR_LOW_HZ, HR_HIGH_HZ, fps_set)
        rr_sig = bandpass_filter(ppg, RR_LOW_HZ, RR_HIGH_HZ, fps_set)

        # Estimate
        hr_raw, hr_conf = estimate_hr_fft(hr_sig, fps_set, 42, 240)
        rr_raw, rr_conf = estimate_hr_fft(rr_sig, fps_set,  9,  24)
        sdnn, rmssd, stress_raw, stress_lbl = compute_stress(hr_sig, fps_set)

        # EMA smoothing
        if hr_raw > 0 and hr_conf > 0.15:
            prev = st.session_state.hr_ema
            st.session_state.hr_ema = (
                EMA_HR * hr_raw + (1-EMA_HR) * prev
                if prev > 0 else hr_raw
            )

        if rr_raw > 0 and rr_conf > 0.12:
            prev = st.session_state.rr_ema
            st.session_state.rr_ema = (
                EMA_RR * rr_raw + (1-EMA_RR) * prev
                if prev > 0 else rr_raw
            )

        if stress_raw > 0:
            prev = st.session_state.stress_ema
            st.session_state.stress_ema = (
                EMA_STRESS * stress_raw + (1-EMA_STRESS) * prev
                if prev > 0 else stress_raw
            )
            st.session_state.stress_label = stress_lbl

        # Display metrics
        hr_val  = st.session_state.hr_ema
        rr_val  = st.session_state.rr_ema
        str_val = st.session_state.stress_ema
        str_lbl = st.session_state.stress_label

        hr_ph.metric(
            "Heart Rate",
            f"{hr_val:.1f} BPM" if hr_val > 0 else "—",
            delta=f"conf {hr_conf:.0%}"
        )
        rr_ph.metric(
            "Resp Rate",
            f"{rr_val:.1f} br/min" if rr_val > 0 else "—"
        )
        str_ph.metric(
            "Stress",
            f"{str_val:.0f}/100" if str_val > 0 else "—",
            delta=str_lbl
        )

        if sdnn > 0:
            hrv_ph.caption(
                f"HRV — SDNN: {sdnn:.1f} ms  |  "
                f"RMSSD: {rmssd:.1f} ms  |  "
                f"Confidence: HR={hr_conf:.0%}  RR={rr_conf:.0%}"
            )

        # History
        if hr_val > 0:
            st.session_state.hr_hist.append(round(hr_val, 1))
            st.session_state.rr_hist.append(round(rr_val, 1) if rr_val > 0 else 0)
            st.session_state.stress_hist.append(round(str_val, 1) if str_val > 0 else 0)
            st.session_state.time_hist.append(round(elapsed, 1))

        # Charts
        if len(st.session_state.hr_hist) > 3:
            t = list(st.session_state.time_hist)
            hr_chart_ph.plotly_chart(
                make_chart(t, list(st.session_state.hr_hist),
                           "rgb(0,210,100)", 40, 180, "BPM"),
                use_container_width=True
            )
            stress_chart_ph.plotly_chart(
                make_chart(t, list(st.session_state.stress_hist),
                           "rgb(255,90,70)", 0, 100, "Stress score"),
                use_container_width=True
            )

    time.sleep(1.0 / fps_set)

cap.release()
st.info("Monitoring stopped. Click Start to begin a new session.")
