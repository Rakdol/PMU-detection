import os
import time
from datetime import datetime, timedelta
from logging import getLogger
import streamlit as st
import pandas as pd
from PIL import Image

from src.db import cruds
from src.configurations import Monitoring, Model
from src.detectors import FrequencyDetector, ROCOFDetector, PcaDetector

from src.data_handler import (
    handle_missing_values,
    save_event_time_data,
)

logger = getLogger(__name__)

st.set_page_config(
    page_title="PMU Monitoring DashBoard",
    layout="wide",
    page_icon="🧊",
    initial_sidebar_state="collapsed",
)

from src.dash.state import session_state_initialize, reset_anomaly_heatmap
from src.dash.metrics import render_metrics, get_style, create_anomaly_metric
from src.dash.chart import (
    plot_line_chart,
    plot_pie_chart,
    plot_anomaly_chart,
    plot_heatmap_chart,
)
from src.dash.side import render_side

session_state_initialize()  # initialize session state
get_style()


model_directory = Model.model_directory
model_file_name = Model.model_file_name
buffer_size = Monitoring.buffer_size

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
timestep = Monitoring.update_step_seconds
KST = Monitoring.KST
plot_line_size = Monitoring.plot_line_size
save_periods_minutes = Monitoring.save_periods_minutes
end_time = end_time = datetime.now() - timedelta(hours=KST)
start_time = end_time - timedelta(seconds=timestep)

while True:
    pmu_data = cruds.select_pmu_from_btw_time(
        st.session_state.db_session,
        start_time=start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
        end_time=end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
        pd_sql=True,
    )

    if pmu_data is None:
        st.error(f"데이터베이스 쿼리 실패: {str(e)}")
        logger.error(f"데이터베이스 쿼리 실패: {str(e)}")
        continue  # 오류 발생 시 루프를 계속 진행

    start_time = end_time
    end_time = start_time + timedelta(seconds=timestep)
    try:
        handle_missing_values(pmu_data)
        pmu_data.index = pd.to_datetime(pmu_data.index) + pd.Timedelta(hours=9)
    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {str(e)}")
        logger.error(f"데이터 처리 중 오류 발생: {str(e)}")
        continue  # 오류 발생 시 루프를 계속 진행

    # 데이터 축적 (이전 데이터와 병합)
    st.session_state.historical_data = pd.concat(
        [st.session_state.historical_data, pmu_data]
    ).tail(buffer_size)

    fr_chart = plot_line_chart(
        st.session_state.historical_data.reset_index().tail(plot_line_size),
        x="timestamp",
        y="Frequency",
        domain=[59.8, 60.2],
    )

    rocof_chart = plot_line_chart(
        st.session_state.historical_data.reset_index().tail(plot_line_size),
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
        render_metrics(st.session_state.historical_data.tail(plot_line_size).mean())

    anomaly_df = pmu_data.copy()

    with anomaly_placeholder.container():
        # Anomaly 그룹
        st.markdown("### Anomaly Detection")
        st.markdown("---")

        st.markdown("#### Frequency & PCA Detector")

        fr_anomalies = fr_detector.detect(
            anomaly_df["Frequency"], st.session_state.fr_threshold
        )

        anomaly_df["FrAnomaly"] = fr_anomalies

        pca_anomailes = pca_detector.process_and_detect(pmu_data.copy(), extreme=True)
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
        # st.write(f"rocof threhold: {st.session_state.rocof_threshold}")

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

        # 파이차트 데이터 생성
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
                create_anomaly_metric(
                    "Current Data Points",
                    st.session_state.daily_detection_data.shape[0],
                )
                create_anomaly_metric(
                    "FR Anomalies",
                    group_counts["FR Anomalies"],
                )

            with m2:

                create_anomaly_metric(
                    "Current No. Anomalies",
                    total_anomailes,
                )

                create_anomaly_metric(
                    "RoCoF Anomalies",
                    group_counts["RoCoF Anomalies"],
                )

            with m3:

                create_anomaly_metric(
                    "Anomaly Percentage",
                    f"{(total_anomailes / st.session_state.daily_detection_data.shape[0]) * 100:.2f}%",
                )

                create_anomaly_metric(
                    "PCA Anomalies",
                    group_counts["PCA Anomalies"],
                )

        with a2:
            p1, p2 = st.columns(2)

            with p1:
                st.altair_chart(global_chart, use_container_width=True)

            with p2:
                st.altair_chart(group_chart, use_container_width=True)

        st.altair_chart(heatmap_chart, use_container_width=True)

    if st.session_state.last_saved_timestamp is None:
        st.session_state.last_saved_timestamp = (
            st.session_state.daily_detection_data.index.min()
        )

    # Get the current max timestamp from the index
    current_timestamp = st.session_state.daily_detection_data.index.max()

    if current_timestamp - st.session_state.last_saved_timestamp >= timedelta(
        minutes=save_periods_minutes
    ):

        df = st.session_state.daily_detection_data[
            (st.session_state.daily_detection_data.index < current_timestamp)
            & (
                st.session_state.daily_detection_data.index
                >= st.session_state.last_saved_timestamp
            )
        ]

        try:
            save_event_time_data(
                df,
                anomaly_col="TotalAnomalies",
                delta_type="seconds",
                delta_time=st.session_state.delta_time,
            )
        except Exception as e:
            st.error(f"이벤트 데이터 저장 실패: {str(e)}")
            logger.error(f"이벤트 데이터 저장 실패: {str(e)}")

        st.session_state.last_saved_timestamp = current_timestamp

    if st.session_state.last_reset_hour != datetime.now().hour:
        st.session_state.last_reset_hour = datetime.now().hour

        st.session_state.daily_detection_data = pd.DataFrame()
        st.session_state.historical_data = pd.DataFrame()
        st.session_state.previous_data = {}
        st.session_state.last_saved_timestamp = None

    if st.session_state.last_reset_date != datetime.now().date():
        st.session_state.last_reset_date = datetime.now().date()

        st.session_state.anomaly_heatmap = reset_anomaly_heatmap()

    time.sleep(timestep)
