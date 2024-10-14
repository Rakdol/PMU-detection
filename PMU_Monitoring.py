import time
from datetime import datetime, timedelta
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from src.db.initialize import initialize_pmu_table
from src.db.database import pdc_engine, SessionPDC
from src.db.models import PmuData
from src.db import cruds

from src.detectors import FrequencyDetector, ROCOFDetector, PcaDetector

initialize_pmu_table(engine=pdc_engine, checkfirst=True)
db = SessionPDC()

# 페이지 설정
st.set_page_config(page_title="Real-time Data Visualization with PMU", layout="wide")
st.sidebar.success("Select a Anomaly page above.")

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

# 이전 데이터 저장용 딕셔너리
previous_data = {}


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
    previous_value = previous_data.get(label)
    if previous_value is None:
        delta = None
    else:
        delta = current_value - previous_value
    previous_data[label] = current_value  # 이전 값 업데이트
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
st.title("📊 Real Time PMU Visualization")
# 차트 업데이트를 위한 공간
chart_placeholder = st.empty()
metric_placeholder = st.empty()

fr_detector = FrequencyDetector()
rocof_detector = ROCOFDetector()


# 실시간 데이터 갱신 루프
historical_data = pd.DataFrame()

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

    # 데이터 축적 (이전 데이터와 병합)
    historical_data = pd.concat([historical_data, pmu_data]).tail(
        100
    )  # 최근 100개의 데이터만 유지

    # # Altair 차트 (선과 이상치 점을 함께 표시)
    line_chart = (
        alt.Chart(historical_data.reset_index())
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", axis=alt.Axis(format="%H:%M:%S")),  # 시간 형식 조정
            y=alt.Y("Frequency:Q", scale=alt.Scale(domain=[59.8, 60.4])),
        )
    )

    line_chart_rocof = (
        alt.Chart(historical_data.reset_index())
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", axis=alt.Axis(format="%H:%M:%S")),  # 시간 형식 조정
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
        st.markdown("### Frequency & DeFrequency")
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
                "Voltage R Angle", f"{current_data['Voltage_R_Angle']:.3f} D", delta
            )
        with col4:
            delta = calculate_delta("Voltage S", current_data["Voltage_S"])
            create_metric("Voltage S", f"{current_data['Voltage_S']:.3f} V", delta)
            delta = calculate_delta("Voltage S Angle", current_data["Voltage_S_Angle"])
            create_metric(
                "Voltage S Angle", f"{current_data['Voltage_S_Angle']:.3f} D", delta
            )
        with col5:
            delta = calculate_delta("Voltage T", current_data["Voltage_T"])
            create_metric("Voltage T", f"{current_data['Voltage_T']:.3f} V", delta)
            delta = calculate_delta("Voltage T Angle", current_data["Voltage_T_Angle"])
            create_metric(
                "Voltage T Angle", f"{current_data['Voltage_T_Angle']:.3f} D", delta
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
                "Current I1 Angle", f"{current_data['Current_I1_Angle']:.4f} D", delta
            )
            delta = calculate_delta("Current I2", current_data["Current_I2"])
            create_metric("Current I2", f"{current_data['Current_I2']:.4f} A", delta)
            delta = calculate_delta(
                "Current I2 Angle", current_data["Current_I2_Angle"]
            )
            create_metric(
                "Current I2 Angle", f"{current_data['Current_I2_Angle']:.4f} D", delta
            )
        with col7:
            delta = calculate_delta("Current I3", current_data["Current_I3"])
            create_metric("Current I3", f"{current_data['Current_I3']:.4f} A", delta)
            delta = calculate_delta(
                "Current I3 Angle", current_data["Current_I3_Angle"]
            )
            create_metric(
                "Current I3 Angle", f"{current_data['Current_I3_Angle']:.4f} D", delta
            )
            delta = calculate_delta("Current I4", current_data["Current_I4"])
            create_metric("Current I4", f"{current_data['Current_I4']:.4f} A", delta)
            delta = calculate_delta(
                "Current I4 Angle", current_data["Current_I4_Angle"]
            )
            create_metric(
                "Current I4 Angle", f"{current_data['Current_I4_Angle']:.4f} D", delta
            )
        with col8:
            delta = calculate_delta("Current I5", current_data["Current_I5"])
            create_metric("Current I5", f"{current_data['Current_I5']:.4f} A", delta)
            delta = calculate_delta(
                "Current I5 Angle", current_data["Current_I5_Angle"]
            )
            create_metric(
                "Current I5 Angle", f"{current_data['Current_I5_Angle']:.4f} D", delta
            )
            delta = calculate_delta("Current I6", current_data["Current_I6"])
            create_metric("Current I6", f"{current_data['Current_I6']:.4f} A", delta)
            delta = calculate_delta(
                "Current I6 Angle", current_data["Current_I6_Angle"]
            )
            create_metric(
                "Current I6 Angle", f"{current_data['Current_I6_Angle']:.4f} D", delta
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
                "Current I7 Angle", f"{current_data['Current_I7_Angle']:.4f} D", delta
            )
        with col10:
            delta = calculate_delta("Current I8", current_data["Current_I8"])
            create_metric("Current I8", f"{current_data['Current_I8']:.4f} A", delta)
            delta = calculate_delta(
                "Current I8 Angle", current_data["Current_I8_Angle"]
            )
            create_metric(
                "Current I8 Angle", f"{current_data['Current_I8_Angle']:.4f} D", delta
            )

    # # 실시간 이상 감지 수행
    # fr_anomalies = fr_detector.detect(
    #     pmu_data["Frequency"].astype(np.float32).values, threshold=0.05
    # )
    # rocof_anomalies = rocof_detector.detect(
    #     pmu_data["DeFrequency"].astype(np.float32).values, threshold=0.0124
    # )

    # # Frequency와 ROCOF의 이상 탐지 결과를 결합
    # frequency_anomalies = fr_anomalies | rocof_anomalies

    # # 이상치 여부를 데이터에 추가
    # pmu_data["Anomaly"] = np.where(frequency_anomalies, pmu_data["Frequency"], np.nan)

    # # 이상치만 표시하는 scatter 차트
    # scatter_chart = (
    #     alt.Chart(pmu_data.reset_index())
    #     .mark_point(size=80, color="red", filled=False)
    #     .encode(x="timestamp:T", y="Anomaly:Q")
    # )

    # # 두 차트 결합
    # combined_chart = line_chart + scatter_chart

    # # Streamlit에 차트 표시
    # with chart_placeholder.container():
    #     st.altair_chart(combined_chart, use_container_width=True)

    # # 이상 감지 결과 메시지 출력
    # if frequency_anomalies.sum() > 0:
    #     st.warning(f"Anomalies Detected: {frequency_anomalies.sum()}")

    # 일정 주기마다 데이터 갱신 (1초 간격)

    time.sleep(1)
