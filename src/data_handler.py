import numpy as np
import pandas as pd
import datetime


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


def save_event_data(
    pmu_data: pd.DataFrame, anomalie_indices: np.ndarray, pad_sequence_length=100
) -> None:
    windows = extract_anomaly_windows(
        anomalie_indices,
        window_size_before=pad_sequence_length,
        window_size_after=pad_sequence_length,
    )
    for i, window in enumerate(windows):
        start = window[0]
        end = window[1]
        start_time = (
            pmu_data["timestamp"][start]
            .replace(" ", "-")
            .replace(":", "-")
            .replace(".", "-")
        )
        end_time = (
            pmu_data["timestamp"][end]
            .replace(" ", "-")
            .replace(":", "-")
            .replace(".", "-")
        )
        print("====== Save Files =======")
        print(f"../event_log/event_data_{start_time}_{end_time}.csv")
        saved_data = pmu_data[start:end]
        saved_data.loc[:, "timestamp"] = pd.to_datetime(saved_data["timestamp"]).apply(
            lambda x: datetime.datetime.strftime(x, "%Y-%m-%d %H:%M:%S:%f")
        )
        saved_data.to_csv(
            f"../event_log/event_data_from_{start_time}_to_{end_time}.csv", index=False
        )
