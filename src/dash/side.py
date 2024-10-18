import streamlit as st
from datetime import datetime


from pathlib import Path


ROOT_PAKAGE = Path(__file__).resolve().parents[2]

model_directory = str(ROOT_PAKAGE / "artifacts")


def render_side(logo, pca_detector):

    with st.sidebar:

        st.image(logo)

        st.title("PMU 데이터 모니터링")
        st.success("감지 기준 설정")

        st.session_state.fr_threshold = st.number_input(
            "주파수 진단 기준",
            value=st.session_state.fr_threshold,
            min_value=0.01,
            max_value=0.3,
            step=0.01,
            format="%.5f",
        )

        st.session_state.rocof_threshold = st.number_input(
            "RoCoF 진단 기준",
            value=st.session_state.rocof_threshold,
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

    with st.sidebar.expander("About the App"):
        st.markdown(
            """
        - 수집되는 PMU 데이터 기반의 데이터 모니터링
        - PMU에서 계측되는 F, V, I 값을 통한 진단 리포트
        - 진단에 필요한 데이터 분석과 진단기 성능 모니터링
        """
        )
