import os
import time
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path


import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image

from src.db.models import PmuData
from src.db import cruds
from src.detectors import FrequencyDetector, ROCOFDetector, PcaDetector
from src.data_handler import (
    handle_missing_values,
    save_event_time_data,
)

PAKAGE_ROOT = Path(__file__).resolve().parents[0]
# PAKAGE_ROOT = "/home/sm/OneDrive/CS/Project/PMU_detection/"

st.set_page_config(
    page_title="PMU Monitoring DashBoard",
    layout="wide",
    page_icon="🧊",
    initial_sidebar_state="collapsed",
)

from src.dash.state import session_state_initialize, reset_anomaly_heatmap
from src.dash.metrics import render_metrics, get_style
from src.dash.chart import (
    plot_line_chart,
    plot_pie_chart,
    plot_anomaly_chart,
    plot_heatmap_chart,
)
from src.dash.side import render_side

session_state_initialize()  # initialize session state
get_style()

model_directory = PAKAGE_ROOT + "artifacts"
model_file_name = "pca_production.pkl"
buffer_size = 60 * 60 * 60


fr_detector = FrequencyDetector()
rocof_detector = ROCOFDetector()
pca_detector = PcaDetector(
    model_directory=model_directory, model_file_name=model_file_name
)

logo = Image.open(os.getcwd() + "/images/logo.png")
new_image = logo.resize((300, 200))

render_side(logo=new_image, pca_detector=pca_detector)

# 실시간 데이터 갱신
st.title("📊 실시간 주파수 변화 현황")
# 차트 업데이트를 위한 공간
chart_placeholder = st.empty()
metric_placeholder = st.empty()
anomaly_placeholder = st.empty()

# # 데이터 스트림 업데이트 루프
timestep = 1
while True:

    # 새로운 데이터 가져오기
    end_time = datetime.now() - timedelta(hours=9)
    start_time = end_time - timedelta(seconds=timestep)

    pmu_data = cruds.select_pmu_from_btw_time(
        st.session_state.db_session,
        start_time=start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        end_time=end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        pd_sql=True,
    )
    handle_missing_values(pmu_data)

    pmu_data.index = pd.to_datetime(pmu_data.index) + pd.Timedelta(hours=9)

    # if "last_timestamp" not in st.session_state:
    last_timestamp = pmu_data.index.min()

    # 데이터 축적 (이전 데이터와 병합)
    st.session_state.historical_data = pd.concat(
        [st.session_state.historical_data, pmu_data]
    ).tail(buffer_size)

    fr_chart = plot_line_chart(
        st.session_state.historical_data.reset_index().tail(60 * 10),
        x="timestamp",
        y="Frequency",
        domain=[59.8, 60.2],
    )

    rocof_chart = plot_line_chart(
        st.session_state.historical_data.reset_index().tail(60 * 10),
        x="timestamp",
        y="DeFrequency",
        domain=[-0.25, 0.25],
    )

    # Streamlit에 차트 및 metric 업데이트
    with chart_placeholder.container():
        st.altair_chart(fr_chart, use_container_width=True)
        st.altair_chart(rocof_chart, use_container_width=True)
        st.markdown("---")

    with metric_placeholder.container():
        render_metrics(st.session_state.historical_data.tail(60 * 10).mean())

    anomaly_df = pmu_data.copy()

    with anomaly_placeholder.container():
        # Anomaly 그룹
        st.markdown("### Anomaly Detection")
        st.markdown("---")

        st.markdown("#### Frequency & PCA Detector")
        fr_anomalies = fr_detector.detect(
            anomaly_df["Frequency"], st.session_state.fr_threshold
        )

        pca_anomailes = pca_detector.process_and_detect(anomaly_df, extreme=True)

        anomaly_df["FrAnomaly"] = fr_anomalies
        anomaly_df["PCAAnomaly"] = pca_anomailes

        fr_chart = plot_anomaly_chart(
            anomaly_df=anomaly_df,
            value_column="Frequency",
            anomaly_columns=["FrAnomaly", "PCAAnomaly"],
            domain=[59.8, 60.2],
        )

        # Streamlit에 차트 출력
        st.altair_chart(fr_chart, use_container_width=True)

        st.markdown("---")
        st.markdown("#### RoCof Detection")
        st.write(f"rocof threhold: {st.session_state.rocof_threshold}")

        rocof_anomailes = rocof_detector.detect(
            anomaly_df["DeFrequency"], threshold=st.session_state.rocof_threshold
        )

        anomaly_df["RoCoFAnomaly"] = rocof_anomailes

        rocof_chart = plot_anomaly_chart(
            anomaly_df=anomaly_df,
            value_column="DeFrequency",
            anomaly_columns=["RoCoFAnomaly"],
            domain=[-0.25, 0.25],
        )

        anomaly_df["TotalAnomalies"] = fr_anomalies | pca_anomailes | rocof_anomailes
        # Streamlit에 차트 출력
        st.altair_chart(rocof_chart, use_container_width=True)

        st.markdown("---")

        # 이상 데이터 축적
        st.session_state.daily_detection_data = pd.concat(
            [st.session_state.daily_detection_data, anomaly_df]
        ).tail(buffer_size)

        total_count = len(st.session_state.daily_detection_data)
        total_anomailes = st.session_state.daily_detection_data["TotalAnomalies"].sum()

        group_counts = {
            "Normal": total_count - total_anomailes,
            "FR Anomalies": st.session_state.daily_detection_data["FrAnomaly"].sum(),
            "RoCoF Anomalies": st.session_state.daily_detection_data[
                "RoCoFAnomaly"
            ].sum(),
            "PCA Anomalies": st.session_state.daily_detection_data["PCAAnomaly"].sum(),
        }

        global_counts = {
            "Normal": total_count - total_anomailes,
            "Total_Anomalies": total_anomailes,
        }

        # 파이차트 데이터프레임으로 변환
        group_df = pd.DataFrame(
            list(group_counts.items()), columns=["Condition", "Count"]
        )

        global_df = pd.DataFrame(
            list(global_counts.items()), columns=["Condition", "Count"]
        )

        # Altair 파이차트 생성
        global_chart = plot_pie_chart(global_df)
        group_chart = plot_pie_chart(group_df)

        heatmap_chart = plot_heatmap_chart(
            st.session_state.anomaly_heatmap,
            st.session_state.daily_detection_data.copy(),
            anomaly_columns=["FrAnomaly", "RoCoFAnomaly", "PCAAnomaly"],
        )

        a1, a2 = st.columns(2)
        with a1:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    "Current Data Points",
                    st.session_state.daily_detection_data.shape[0],
                )
                st.metric("FR Anomalies", group_counts["FR Anomalies"])
            with m2:
                st.metric("Current No. Anomalies", total_anomailes)
                st.metric("RoCoF Anomalies", group_counts["RoCoF Anomalies"])
            with m3:
                st.metric(
                    "Anomaly Percentage",
                    f"{(total_anomailes / st.session_state.daily_detection_data.shape[0]) * 100:.2f}%",
                )
                st.metric("PCA Anomalies", group_counts["PCA Anomalies"])

        with a2:
            p1, p2 = st.columns(2)

            with p1:
                st.altair_chart(global_chart, use_container_width=True)

            with p2:
                st.altair_chart(group_chart, use_container_width=True)

        st.altair_chart(heatmap_chart, use_container_width=True)

    if "last_saved_timestamp" not in st.session_state:
        st.session_state.last_saved_timestamp = (
            st.session_state.daily_detection_data.index.min()
        )
    # Get the current max timestamp from the index
    current_timestamp = st.session_state.daily_detection_data.index.max()

    if current_timestamp - st.session_state.last_saved_timestamp >= timedelta(
        minutes=1
    ):

        min_time_index = st.session_state.last_saved_timestamp - timedelta(minutes=1)
        df = st.session_state.daily_detection_data[
            st.session_state.daily_detection_data.index >= min_time_index
        ]

        save_event_time_data(
            df,
            anomaly_col="TotalAnomalies",
            delta_type="seconds",
            delta_time=30,
        )

        # st.dataframe(x)
        st.session_state.last_saved_timestamp = current_timestamp

    if st.session_state.last_reset_date != datetime.now().date():
        st.session_state.last_reset_date = datetime.now().date()
        st.session_state.anomaly_heatmap = reset_anomaly_heatmap()

        st.session_state.daily_detection_data = pd.DataFrame()
        st.session_state.historical_data = pd.DataFrame()
        st.session_state.previous_data = {}

    time.sleep(timestep)
