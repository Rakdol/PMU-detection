import streamlit as st
import pandas as pd

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
