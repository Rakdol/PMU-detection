import streamlit as st


def get_style():

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


# 실시간 데이터 갱신 (이전 값과 현재 값을 비교하여 delta 계산)
def calculate_delta(label, current_value):
    previous_value = st.session_state.previous_data.get(label)
    if previous_value is None:
        delta = None
    else:
        delta = current_value - previous_value
    st.session_state.previous_data[label] = current_value  # 이전 값 업데이트
    return delta


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


def render_metrics(placeholder, current_data):
    # 현재 상태를 metric으로 표
    with placeholder.container():
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
