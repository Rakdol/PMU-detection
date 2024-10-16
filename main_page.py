import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image

from src.db.models import PmuData
from src.db import cruds
from src.detectors import FrequencyDetector, ROCOFDetector, PcaDetector

# PAKAEG_ROOT = Path(__file__).resolve().parents[0]
PAKAGE_ROOT = "/home/sm/OneDrive/CS/Project/PMU_detection/"

st.set_page_config(
    page_title="PMU Monitoring DashBoard",
    layout="wide",
    page_icon="🧊",
    initial_sidebar_state="collapsed",
)

from src.dash.state import session_state_initialize
from src.dash.metrics import render_metrics, get_style
from src.dash.chart import plot_line_chart

session_state_initialize()  # initialize session state
get_style()

model_directory = PAKAGE_ROOT + "artifacts"
model_file_name = "pca_production.pkl"

fr_detector = FrequencyDetector()
rocof_detector = ROCOFDetector()
pca_detector = PcaDetector(
    model_directory=model_directory, model_file_name=model_file_name
)

with st.sidebar:
    logo = Image.open(os.getcwd() + "/images/logo.png")
    new_image = logo.resize((300, 400))
    st.image(logo)

    st.title("PMU 데이터 모니터링")
    st.success("감지 기준 설정")

    fr_threshold = st.number_input(
        "주파수 진단 기준",
        value=0.2,
        min_value=0.01,
        max_value=0.3,
        step=0.01,
        format="%.5f",
    )

    rocof_threshold = st.number_input(
        "RoCoF 진단 기준",
        value=0.0124,
        min_value=0.001,
        max_value=0.3,
        step=0.001,
        format="%.5f",
    )

    if st.button("Save Pca Model"):
        timeindex = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        file_name = f"pca_{timeindex}.pkl"
        pca_detector.save_model(
            model_directory=model_directory, model_file_name=file_name
        )


# 실시간 데이터 갱신
st.title("📊 실시간 주파수 변화 현황")
# 차트 업데이트를 위한 공간
chart_placeholder = st.empty()
metric_placeholder = st.empty()
anomaly_placeholder = st.empty()

# 실시간 데이터 갱신 루프
historical_data = st.session_state.historical_data
daily_detection_data = st.session_state.daily_detection_data

# # 데이터 스트림 업데이트 루프
while True:

    # 새로운 데이터 가져오기
    end_time = datetime.now() - timedelta(hours=9)
    start_time = end_time - timedelta(seconds=1)

    pmu_data = cruds.select_pmu_from_btw_time(
        st.session_state.db_session,
        start_time=start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        end_time=end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        pd_sql=True,
    )
    pmu_data.index = pd.to_datetime(pmu_data.index) + pd.Timedelta(hours=9)

    # 데이터 축적 (이전 데이터와 병합)
    historical_data = pd.concat([historical_data, pmu_data]).tail(1000)

    fr_chart = plot_line_chart(
        historical_data.reset_index(), x="timestamp", y="Frequency", domain=[59.8, 60.4]
    )

    rocof_chart = plot_line_chart(
        historical_data.reset_index(),
        x="timestamp",
        y="DeFrequency",
        domain=[-0.25, 0.25],
    )

    # Streamlit에 차트 및 metric 업데이트
    with chart_placeholder.container():
        st.altair_chart(fr_chart, use_container_width=True)
        st.altair_chart(rocof_chart, use_container_width=True)
        st.markdown("---")

    render_metrics(metric_placeholder, historical_data.mean())

    with anomaly_placeholder.container():

        # Anomaly 그룹
        st.markdown("### Anomaly Detection")
        col11, col12 = st.columns(2)

        with col11:
            st.markdown("#### Frequency Detector")
            fr_anomalies = fr_detector.detect(
                historical_data["Frequency"], fr_threshold
            )

            historical_data["FrAnomaly"] = fr_anomalies
            # Altair 차트 생성
            base = alt.Chart(historical_data["Frequency"]).encode(x="timestamp:T")

            # 실수 데이터를 라인 차트로 시각화
            line = base.mark_line().encode(
                y="Frequency:Q",
                color=alt.value("blue"),
                tooltip=["timestamp:T", "Frequency:Q"],
            )

            # 이상값(True)을 기준으로 마커 표시
            anomalies = (
                base.mark_point(color="red", size=100)
                .encode(y="Frequency:Q", tooltip=["timestamp:T", "Frequency:Q"])
                .transform_filter(alt.datum.FrAnomaly == True)  # True인 경우만 필터링
            )

            # 차트 결합
            chart = line + anomalies

            # Streamlit에 차트 출력
            st.altair_chart(chart, use_container_width=True)

        with col12:
            st.markdown("#### RoCof Detection")
            st.write(f"rocof threhold: {rocof_threshold}")
            rocof_anomailes = rocof_detector.detect(
                historical_data["DeFrequency"], threshold=rocof_threshold
            )
            st.write(rocof_anomailes)

        st.markdown("#### PCA Detection")
        pca_anomailes = pca_detector.process_and_detect(historical_data)

        st.write(
            pd.DataFrame(
                pca_anomailes,
                index=historical_data.index,
                columns=["pca_anomailes"],
            )
        )
        total_anomailes = fr_anomalies | rocof_anomailes | pca_anomailes

        # 이상 감지 비율 및 기본 통계 제공
        st.metric("Total Data Points", historical_data.shape[0])
        st.metric("Total Anomalies", sum(total_anomailes))
        st.metric(
            "Anomaly Percentage",
            f"{(sum(total_anomailes) / historical_data.shape[0]) * 100:.2f}%",
        )

    time.sleep(1)
