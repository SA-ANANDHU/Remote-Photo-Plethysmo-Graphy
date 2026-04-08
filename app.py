"""
dashboard/app.py
=================
What this file does:
  - Streamlit web dashboard for real-time vitals display
  - Live webcam feed with face detection overlay
  - Real-time heart rate, respiratory rate, stress charts
  - Session history and CSV export

How to run:
  streamlit run dashboard/app.py

Interview explanation:
  "I built the dashboard in Streamlit because it lets me
   deploy an ML demo as a web app in Python-only code —
   no JavaScript, no React. For a production system I would
   use FastAPI + a proper frontend, but for portfolio
   demonstration Streamlit shows the full pipeline running
   end-to-end with a professional UI in minimal time."
"""

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from collections import deque
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.signal_extractor import FaceROIExtractor, SignalBuffer
from preprocessing.filters import (bandpass_filter, chrom_rppg,
                                    quality_weighted_filter)
from vitals.heart_rate import HeartRateEstimator
from vitals.resp_rate import RespiratoryRateEstimator
from vitals.stress_index import StressEstimator


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="rPPG Vitals Engine",
    page_icon="❤",
    layout="wide"
)

st.title("rPPG Vitals Engine")
st.caption("Contactless heart rate, respiratory rate & stress from webcam · No wearable needed")


# ─────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────
if 'running' not in st.session_state:
    st.session_state.running = False
if 'hr_history' not in st.session_state:
    st.session_state.hr_history = deque(maxlen=300)
if 'rr_history' not in st.session_state:
    st.session_state.rr_history = deque(maxlen=300)
if 'stress_history' not in st.session_state:
    st.session_state.stress_history = deque(maxlen=300)
if 'time_history' not in st.session_state:
    st.session_state.time_history = deque(maxlen=300)


# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    start_btn = st.button("Start Monitoring",
                          type="primary",
                          use_container_width=True)
    stop_btn  = st.button("Stop",
                          use_container_width=True)

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    st.divider()
    st.subheader("Settings")
    fps_setting = st.slider("Camera FPS", 15, 60, 30, step=5)
    window_sec  = st.slider("Signal window (seconds)", 10, 60, 30, step=5)

    st.divider()
    st.subheader("Export")
    if st.button("Export session to CSV", use_container_width=True):
        if len(st.session_state.hr_history) > 0:
            df = pd.DataFrame({
                'time_s':     list(st.session_state.time_history),
                'heart_rate': list(st.session_state.hr_history),
                'resp_rate':  list(st.session_state.rr_history),
                'stress':     list(st.session_state.stress_history)
            })
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="vitals_session.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.divider()
    st.caption("rPPG Vitals Engine v1.0")
    st.caption("Anandhu — Final Year Project")


# ─────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────
col_cam, col_vitals = st.columns([1, 1])

with col_cam:
    st.subheader("Live feed")
    cam_placeholder = st.empty()

with col_vitals:
    st.subheader("Current readings")
    metric_row = st.columns(3)
    hr_metric     = metric_row[0].empty()
    rr_metric     = metric_row[1].empty()
    stress_metric = metric_row[2].empty()

    progress_label = st.empty()
    progress_bar   = st.empty()

st.divider()

col_hr_chart, col_rr_chart = st.columns(2)
with col_hr_chart:
    st.subheader("Heart rate trend")
    hr_chart = st.empty()

with col_rr_chart:
    st.subheader("Stress level trend")
    stress_chart = st.empty()


# ─────────────────────────────────────────────
# Helper: build Plotly line chart
# ─────────────────────────────────────────────
def make_line_chart(times, values, label, color, y_min, y_max):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(times),
        y=list(values),
        mode='lines',
        name=label,
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.1)')
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(range=[y_min, y_max]),
        xaxis_title="Time (s)",
        yaxis_title=label,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# ─────────────────────────────────────────────
# Main monitoring loop
# ─────────────────────────────────────────────
if st.session_state.running:

    extractor  = FaceROIExtractor()
    buffer     = SignalBuffer(fps=fps_setting, window_seconds=window_sec)
    hr_est     = HeartRateEstimator(fps=fps_setting)
    rr_est     = RespiratoryRateEstimator(fps=fps_setting)
    stress_est = StressEstimator(fps=fps_setting)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam. Check camera permissions.")
        st.session_state.running = False
        st.stop()

    start_time = time.time()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        elapsed = time.time() - start_time

        # Extract signal
        sample, roi_box = extractor.extract(frame)

        if sample is not None:
            buffer.add(sample, elapsed)
            if roi_box:
                x1, y1, x2, y2 = roi_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

        # Display camera frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Update progress
        pct = min(100, int(buffer.seconds_collected / window_sec * 100))
        progress_label.caption(f"Signal buffer: {buffer.seconds_collected:.0f}s / {window_sec}s")
        progress_bar.progress(pct / 100)

        # Compute vitals when enough signal
        if buffer.is_ready(min_seconds=10.0):
            arrays  = buffer.get_arrays()
            r, g, b = arrays['r'], arrays['g'], arrays['b']
            quality = arrays['quality']

            # CHROM + filter
            ppg_raw = chrom_rppg(r, g, b)
            ppg_raw = quality_weighted_filter(ppg_raw, quality)
            hr_sig  = bandpass_filter(ppg_raw, 0.7, 4.0, fps_setting)
            rr_sig  = bandpass_filter(ppg_raw, 0.15, 0.4, fps_setting)

            hr_result     = hr_est.estimate(hr_sig)
            rr_result     = rr_est.estimate(rr_sig)
            stress_result = stress_est.estimate(hr_sig)

            # Update metrics
            hr_val     = hr_result['hr_bpm']
            rr_val     = rr_result['rr_bpm']
            stress_val = stress_result['stress_score']
            stress_lbl = stress_result['stress_label']

            hr_display     = f"{hr_val:.1f}" if hr_val > 0 else "—"
            rr_display     = f"{rr_val:.1f}" if rr_val > 0 else "—"
            stress_display = f"{stress_val:.0f}" if stress_val > 0 else "—"

            hr_metric.metric("Heart Rate", f"{hr_display} BPM",
                             delta=None)
            rr_metric.metric("Resp Rate", f"{rr_display} br/min")
            stress_metric.metric("Stress", f"{stress_display}/100",
                                 delta=stress_lbl)

            # Append to history
            if hr_val > 0:
                st.session_state.hr_history.append(hr_val)
                st.session_state.rr_history.append(rr_val if rr_val > 0 else 0)
                st.session_state.stress_history.append(stress_val if stress_val > 0 else 0)
                st.session_state.time_history.append(elapsed)

            # Update charts
            if len(st.session_state.hr_history) > 2:
                times = list(st.session_state.time_history)
                hr_chart.plotly_chart(
                    make_line_chart(times,
                                    list(st.session_state.hr_history),
                                    "HR (BPM)", "rgb(0,200,100)", 40, 180),
                    use_container_width=True
                )
                stress_chart.plotly_chart(
                    make_line_chart(times,
                                    list(st.session_state.stress_history),
                                    "Stress", "rgb(255,100,80)", 0, 100),
                    use_container_width=True
                )

        time.sleep(1.0 / fps_setting)

    cap.release()

else:
    cam_placeholder.info("Click 'Start Monitoring' in the sidebar to begin.")
    hr_metric[0].metric("Heart Rate", "— BPM") if hasattr(hr_metric, '__getitem__') else None
