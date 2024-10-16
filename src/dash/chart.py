import altair as alt


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


# def chart_streaming():
#     line
