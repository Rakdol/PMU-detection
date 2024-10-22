import altair as alt
import pandas as pd


# Altair 차트 설정
def plot_line_chart(data, x: str, y: str, domain: list):
    line_chart = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X(x, axis=alt.Axis(format="%Y-%m-%d %H:%M:%S.%f")),
            y=alt.Y(y, scale=alt.Scale(domain=domain)),
        )
    )

    return line_chart


def plot_pie_chart(df):
    # Altair 파이차트 생성
    pie_chart = (
        alt.Chart(df)
        .mark_arc()
        .encode(
            theta=alt.Theta(field="Count", type="quantitative"),  # 값의 비율
            color=alt.Color(field="Condition", type="nominal"),  # 카테고리 색상
            tooltip=["Condition", "Count"],  # 툴팁 설정
        )
        .properties(width=400, height=400)
    )

    return pie_chart


def plot_anomaly_chart(
    anomaly_df: pd.DataFrame,
    value_column: str,
    anomaly_columns: list[str],
    domain=list,
):

    cols = ["timestamp"] + [value_column] + anomaly_columns

    data = anomaly_df.copy().reset_index().loc[:, cols]

    # Altair 차트 생성
    base = alt.Chart(data).encode(
        alt.X("timestamp:T", axis=alt.Axis(format="%Y-%m-%d %H:%M:%S.%f"))
    )

    # 실수 데이터를 라인 차트로 시각화
    line = base.mark_line().encode(
        y=alt.Y(f"{value_column}:Q", scale=alt.Scale(domain=domain)),
        color=alt.value("blue"),
        tooltip=["timestamp:T", f"{value_column}:Q"],
    )

    for a_col in anomaly_columns:
        anomalies = (
            base.mark_point(size=100)
            .encode(y=f"{value_column}:Q", tooltip=["timestamp:T", f"{value_column}:Q"])
            .transform_filter(alt.datum[a_col] == True)  # True인 경우만 필터링
        )
        line += anomalies

    return line


def plot_heatmap_chart(
    heatmap: pd.DataFrame,
    data: pd.DataFrame,
    anomaly_columns: list[str] = ["fr_anomalies", "rocof_anomalies", "pca_anomalies"],
):

    data = data.reset_index()
    data["anomalies"] = data.loc[:, anomaly_columns].sum(axis=1)
    data["minute"] = pd.to_datetime(data["timestamp"]).dt.minute
    data["hour"] = pd.to_datetime(data["timestamp"]).dt.hour

    minute_anomalies = data.groupby(["hour", "minute"])["anomalies"].sum().reset_index()

    # 기본 시간대에 이상 데이터를 병합 (없는 시간대는 0으로 처리)
    full_data = pd.merge(heatmap, minute_anomalies, on=["hour", "minute"], how="left")
    full_data["anomalies"] = full_data["anomalies"].fillna(0)  # NaN을 0으로 처리

    heatmap = (
        alt.Chart(full_data)
        .mark_rect()
        .encode(
            x=alt.X("minute:O", title="Minute of the hour"),
            y=alt.Y("hour:O", title="Hour of the day"),
            color=alt.Color(
                "anomalies:Q", title="Anomaly Count", scale=alt.Scale(scheme="reds")
            ),
            tooltip=["hour", "minute", "anomalies"],
        )
        .properties(width=600, height=400, title="Anomaly Count Heatmap")
    )

    return heatmap
