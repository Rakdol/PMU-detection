from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

from src.db.initialize import initialize_pmu_table
from src.db.database import pdc_engine, SessionPDC


def session_state_initialize():

    # db session init
    if "db_session" not in st.session_state:
        initialize_pmu_table(engine=pdc_engine, checkfirst=True)
        st.session_state.db_session = SessionPDC()

    # 이전 데이터 저장용 딕셔너리 및 historical_data 초기화
    if "previous_data" not in st.session_state:
        st.session_state.previous_data = {}

    if "historical_data" not in st.session_state:
        st.session_state.historical_data = pd.DataFrame()

    if "daily_detection_data" not in st.session_state:
        st.session_state.daily_detection_data = pd.DataFrame()

    if "fr_threshold" not in st.session_state:
        st.session_state.fr_threshold = 0.2

    if "rocof_threshold" not in st.session_state:
        st.session_state.rocof_threshold = 0.0124

    if "anomaly_heatmap" not in st.session_state:
        # 모든 시간대와 분을 포함한 기본 데이터프레임 생성
        hours = np.arange(0, 24)
        minutes = np.arange(0, 60)
        anomaly_heatmap = pd.DataFrame(
            [(h, m) for h in hours for m in minutes], columns=["hour", "minute"]
        )
        st.session_state.anomaly_heatmap = anomaly_heatmap

    if "last_reset_hour" not in st.session_state:
        st.session_state.last_reset_hour = datetime.now().hour

    if "last_reset_date" not in st.session_state:
        st.session_state.last_reset_date = datetime.now().date()

    if "last_saved_timestamp" not in st.session_state:
        st.session_state.last_saved_timestamp = None

    if "delta_time" not in st.session_state:
        st.session_state.delta_time = 30


def reset_anomaly_heatmap():
    hours = np.arange(0, 24)
    minutes = np.arange(0, 60)
    anomaly_heatmap = pd.DataFrame(
        [(h, m) for h in hours for m in minutes], columns=["hour", "minute"]
    )
    st.session_state.anomaly_heatmap = anomaly_heatmap
