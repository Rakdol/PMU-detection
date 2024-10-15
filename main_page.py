import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image

from src.db.initialize import initialize_pmu_table
from src.db.database import pdc_engine, SessionPDC
from src.db.models import PmuData
from src.db import cruds

from src.detectors import FrequencyDetector, ROCOFDetector, PcaDetector

PAKAEG_ROOT = Path(__file__).resolve().parents[0]

st.set_page_config(
    page_title="PMU Monitoring DashBoard",
    layout="wide",
    page_icon="🧊",
    initial_sidebar_state="collapsed",
)


if "db_session" not in st.session_state:
    initialize_pmu_table(engine=pdc_engine, checkfirst=True)
    st.session_state.db_session = SessionPDC()

db = st.session_state.db_session

model_directory = str(PAKAEG_ROOT / "artifacts")
model_file_name = ""

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


# CSS 스타일 정의 - 어두운 테마 적용
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.6);
        margin-bottom: 15px;
        color: white;
    }
    .metric-label {
        font-size: 20px;
        font-weight: bold;
        color: #f5f5f5;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1DB954;  /* Green color for positive values */
    }
    .metric-delta-positive {
        font-size: 18px;
        color: #1DB954; /* Green for positive deltas */
    }
    .metric-delta-negative {
        font-size: 18px;
        color: #FF6347; /* Red for negative deltas */
    }
    .metric-delta-neutral {
        font-size: 18px;
        color: #f5f5f5; /* Neutral color for no change */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 이전 데이터 저장용 딕셔너리 및 historical_data 초기화
if "previous_data" not in st.session_state:
    st.session_state.previous_data = {}

if "historical_data" not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()

if "daily_detection_data" not in st.session_state:
    st.session_state.daily_abnormal_data = pd.DataFrame()


# HTML을 활용한 메트릭 카드 생성
def create_metric(label, value, delta=None):
    delta_class = "metric-delta-neutral"
    if delta is not None:
        if delta > 0:
            delta_class = "metric-delta-positive"
        elif delta < 0:
            delta_class = "metric-delta-negative"

    metric_html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        """

    if delta is not None:
        metric_html += f'<div class="{delta_class}">Δ {delta:.3f}</div>'

    metric_html += "</div>"

    st.markdown(metric_html, unsafe_allow_html=True)


# 실시간 데이터 갱신 (이전 값과 현재 값을 비교하여 delta 계산)
def calculate_delta(label, current_value):
    previous_value = st.session_state.previous_data.get(label)
    if previous_value is None:
        delta = None
    else:
        delta = current_value - previous_value
    st.session_state.previous_data[label] = current_value  # 이전 값 업데이트
    return delta


# Altair 차트 설정
def plot_chart(data):
    # Altair 차트 (부드러운 시각화를 위해 interpolate 사용)
    line_chart = (
        alt.Chart(data)
        .mark_line(interpolate="monotone")
        .encode(x="timestamp:T", y="Frequency:Q")
        .properties(width=700, height=400)
    )
    return line_chart


# 실시간 데이터 갱신
st.title("📊 실시간 주파수 변화 현황")
# 차트 업데이트를 위한 공간
chart_placeholder = st.empty()
metric_placeholder = st.empty()
anomaly_placeholder = st.empty()

# 실시간 데이터 갱신 루프
historical_data = st.session_state.historical_data

# # 데이터 스트림 업데이트 루프
while True:
    # 새로운 데이터 가져오기
    end_time = datetime.now() - timedelta(hours=9)
    start_time = end_time - timedelta(seconds=1)

    pmu_data = cruds.select_pmu_from_btw_time(
        db,
        start_time=start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        end_time=end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        pd_sql=True,
    )
    pmu_data.index = pd.to_datetime(pmu_data.index) + pd.Timedelta(hours=9)

    # 데이터 축적 (이전 데이터와 병합)
    historical_data = pd.concat([historical_data, pmu_data]).tail(1000)

    line_chart = (
        alt.Chart(historical_data.reset_index())
        .mark_line()
        .encode(
            x=alt.X(
                "timestamp:T", axis=alt.Axis(format="%Y-%m-%d %H:%M:%S")
            ),  # 시간 형식 조정
            y=alt.Y("Frequency:Q", scale=alt.Scale(domain=[59.8, 60.4])),
        )
    )

    line_chart_rocof = (
        alt.Chart(historical_data.reset_index())
        .mark_line()
        .encode(
            x=alt.X(
                "timestamp:T", axis=alt.Axis(format="%Y-%m-%d %H:%M:%S")
            ),  # 시간 형식 조정
            y=alt.Y("DeFrequency:Q", scale=alt.Scale(domain=[-0.25, 0.25])),
        )
    )

    # Streamlit에 차트 및 metric 업데이트
    with chart_placeholder.container():
        st.altair_chart(line_chart, use_container_width=True)
        st.altair_chart(line_chart_rocof, use_container_width=True)

    # 현재 상태를 metric으로 표시
    current_data = historical_data.mean()  # 가장 최근 데이터를 가져옴
    with metric_placeholder.container():
        # Frequency와 DeFrequency 그룹
        st.markdown("### 주파수 & 주파수 변화율")
        col1, col2 = st.columns(2)
        with col1:
            delta = calculate_delta("Current Frequency", current_data["Frequency"])
            create_metric(
                "Current Frequency", f"{current_data['Frequency']:.3f} Hz", delta
            )
        with col2:
            delta = calculate_delta("Current DeFrequency", current_data["DeFrequency"])
            create_metric(
                "Current DeFrequency", f"{current_data['DeFrequency']:.6f} Hz/s", delta
            )

        st.markdown("---")

        # Voltage 그룹
        st.markdown("### Voltage Measurements")
        col3, col4, col5 = st.columns(3)
        with col3:
            delta = calculate_delta("Voltage R", current_data["Voltage_R"])
            create_metric("Voltage R", f"{current_data['Voltage_R']:.3f} V", delta)
            delta = calculate_delta("Voltage R Angle", current_data["Voltage_R_Angle"])
            create_metric(
                "Voltage R Angle", f"{current_data['Voltage_R_Angle']:.3f} °", delta
            )
        with col4:
            delta = calculate_delta("Voltage S", current_data["Voltage_S"])
            create_metric("Voltage S", f"{current_data['Voltage_S']:.3f} V", delta)
            delta = calculate_delta("Voltage S Angle", current_data["Voltage_S_Angle"])
            create_metric(
                "Voltage S Angle", f"{current_data['Voltage_S_Angle']:.3f} °", delta
            )
        with col5:
            delta = calculate_delta("Voltage T", current_data["Voltage_T"])
            create_metric("Voltage T", f"{current_data['Voltage_T']:.3f} V", delta)
            delta = calculate_delta("Voltage T Angle", current_data["Voltage_T_Angle"])
            create_metric(
                "Voltage T Angle", f"{current_data['Voltage_T_Angle']:.3f} °", delta
            )

        st.markdown("---")

        # Current 그룹
        st.markdown("### Current Measurements")
        col6, col7, col8 = st.columns(3)
        with col6:
            delta = calculate_delta("Current I1", current_data["Current_I1"])
            create_metric("Current I1", f"{current_data['Current_I1']:.4f} A", delta)
            delta = calculate_delta(
                "Current I1 Angle", current_data["Current_I1_Angle"]
            )
            create_metric(
                "Current I1 Angle", f"{current_data['Current_I1_Angle']:.4f} °", delta
            )
            delta = calculate_delta("Current I2", current_data["Current_I2"])
            create_metric("Current I2", f"{current_data['Current_I2']:.4f} A", delta)
            delta = calculate_delta(
                "Current I2 Angle", current_data["Current_I2_Angle"]
            )
            create_metric(
                "Current I2 Angle", f"{current_data['Current_I2_Angle']:.4f} °", delta
            )
        with col7:
            delta = calculate_delta("Current I3", current_data["Current_I3"])
            create_metric("Current I3", f"{current_data['Current_I3']:.4f} A", delta)
            delta = calculate_delta(
                "Current I3 Angle", current_data["Current_I3_Angle"]
            )
            create_metric(
                "Current I3 Angle", f"{current_data['Current_I3_Angle']:.4f} °", delta
            )
            delta = calculate_delta("Current I4", current_data["Current_I4"])
            create_metric("Current I4", f"{current_data['Current_I4']:.4f} A", delta)
            delta = calculate_delta(
                "Current I4 Angle", current_data["Current_I4_Angle"]
            )
            create_metric(
                "Current I4 Angle", f"{current_data['Current_I4_Angle']:.4f} °", delta
            )
        with col8:
            delta = calculate_delta("Current I5", current_data["Current_I5"])
            create_metric("Current I5", f"{current_data['Current_I5']:.4f} A", delta)
            delta = calculate_delta(
                "Current I5 Angle", current_data["Current_I5_Angle"]
            )
            create_metric(
                "Current I5 Angle", f"{current_data['Current_I5_Angle']:.4f} °", delta
            )
            delta = calculate_delta("Current I6", current_data["Current_I6"])
            create_metric("Current I6", f"{current_data['Current_I6']:.4f} A", delta)
            delta = calculate_delta(
                "Current I6 Angle", current_data["Current_I6_Angle"]
            )
            create_metric(
                "Current I6 Angle", f"{current_data['Current_I6_Angle']:.4f} °", delta
            )

        # Current 그룹 2 (I7, I8)
        col9, col10 = st.columns(2)
        with col9:
            delta = calculate_delta("Current I7", current_data["Current_I7"])
            create_metric("Current I7", f"{current_data['Current_I7']:.4f} A", delta)
            delta = calculate_delta(
                "Current I7 Angle", current_data["Current_I7_Angle"]
            )
            create_metric(
                "Current I7 Angle", f"{current_data['Current_I7_Angle']:.4f} °", delta
            )
        with col10:
            delta = calculate_delta("Current I8", current_data["Current_I8"])
            create_metric("Current I8", f"{current_data['Current_I8']:.4f} A", delta)
            delta = calculate_delta(
                "Current I8 Angle", current_data["Current_I8_Angle"]
            )
            create_metric(
                "Current I8 Angle", f"{current_data['Current_I8_Angle']:.4f} °", delta
            )

        st.markdown("---")

        with anomaly_placeholder.container():

            # Anomaly 그룹
            st.markdown("### Anomaly Detection")
            col11, col12 = st.columns(2)

            with col11:
                st.markdown("#### Frequency Detector")
                fr_anomalies = fr_detector.detect(
                    historical_data["Frequency"], fr_threshold
                )

                st.write(fr_anomalies)

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
