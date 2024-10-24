from io import BytesIO
from datetime import timedelta
from pathlib import Path
from logging import getLogger
import numpy as np
import pandas as pd


from src.bucket import client, bucket_name

logger = getLogger(__name__)

PAKAGE_ROOT = Path(__file__).resolve().parents[1]
EVENT_PATH = str(PAKAGE_ROOT / "event_logs")


def handle_missing_values(missing_data_frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values."""
    return missing_data_frame.ffill().bfill()


def extract_anomaly_windows(
    anomaly_indices: np.ndarray, window_size_before: int, window_size_after: int
) -> list:

    anomalies = np.sort(anomaly_indices)

    window = []

    start = max(0, anomalies[0] - window_size_before)
    end = anomalies[0] + window_size_after

    for index in anomalies[1:]:
        new_start = max(0, index - window_size_before)
        new_end = index + window_size_after

        if new_start <= end:
            end = max(end, new_end)
        else:
            window.append((start, end))
            start = new_start
            end = new_end

    window.append((start, end))

    return window


def extract_window_timedelta(
    df: pd.DataFrame, anomaly_col: str, delta_type: str, delta_time: int
) -> list[tuple]:

    window = []
    indices = df.index[df[anomaly_col] > 0]

    if "seconds" == delta_type:
        td = timedelta(seconds=delta_time)

    elif "minutes" == delta_type:
        td = timedelta(minutes=delta_time)

    else:
        td = timedelta(seconds=10)

    start = max(
        df.index.min(),
        indices[0] - td,
    )

    end = min(
        df.index.max(),
        indices[0] + td,
    )

    for index in indices[1:]:
        new_start = max(
            df.index.min(),
            index - td,
        )

        new_end = min(
            df.index.max(),
            index + td,
        )
        if new_start <= end:
            end = max(end, new_end)
        else:
            print(start, end)
            window.append((start, end))
            start = new_start
            end = new_end

    window.append((start, end))
    return window


def save_event_time_data(
    event_data: pd.DataFrame, anomaly_col: str, delta_type: str, delta_time: int
):

    windows = extract_window_timedelta(
        df=event_data,
        anomaly_col=anomaly_col,
        delta_type=delta_type,
        delta_time=delta_time,
    )

    max_time = event_data.index.max()

    for i, window in enumerate(windows):
        start = max(event_data.index.min(), window[0])
        end = min(window[1], max_time)

        start_time = start.strftime("%Y-%m-%d-%H-%M-%S-%f")
        end_time = end.strftime("%Y-%m-%d-%H-%M-%S-%f")
        # 저장 파일 이름 출력
        logger.info("====== Save Files =======")
        save_directory = EVENT_PATH + f"/event_data_{start_time}_{end_time}.csv"
        logger.info(save_directory)
        # 데이터 추출
        saved_data = event_data.loc[start:end]
        saved_data = saved_data[~saved_data.index.duplicated(keep="first")]
        csv = saved_data.to_csv().encode("utf-8")

        # csv to bytes, upload minio
        result = client.put_object(
            bucket_name, save_directory, BytesIO(csv), len(saved_data)
        )
        logger.info(f"data upload to MinIO - {result}")

        # 데이터 저장


def save_event_data(
    pmu_data: pd.DataFrame, anomalie_indices: np.ndarray, pad_sequence_length=1000
) -> None:
    windows = extract_anomaly_windows(
        anomalie_indices,
        window_size_before=pad_sequence_length,
        window_size_after=pad_sequence_length,
    )

    data_len = len(pmu_data)

    for i, window in enumerate(windows):
        # Start와 End 값을 데이터 범위 내로 제한
        start = max(0, window[0])
        end = min(window[1], data_len - 1)

        # 시작과 끝의 타임스탬프를 가져와 파일 이름으로 사용
        start_time = pd.to_datetime(pmu_data["timestamp"].iloc[start]).strftime(
            "%Y-%m-%d-%H-%M-%S-%f"
        )
        end_time = pd.to_datetime(pmu_data["timestamp"].iloc[end]).strftime(
            "%Y-%m-%d-%H-%M-%S-%f"
        )
        # 저장 파일 이름 출력
        print("====== Save Files =======")

        save_directory = EVENT_PATH + f"/event_data_{start_time}_{end_time}.csv"
        print(save_directory)

        # 데이터 추출
        saved_data = pmu_data.loc[start : end + 1]  # end가 포함되도록 +1
        saved_data["timestamp"] = pd.to_datetime(saved_data["timestamp"]).apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S.%f")
        )

        # 데이터 저장
        saved_data.to_csv(save_directory, index=False)
