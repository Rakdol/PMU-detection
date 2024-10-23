from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]


class Monitoring:
    update_step_seconds = 1
    buffer_size = 60 * 60 * 60  # 1초 60 샘플 - 1시간
    KST = 9
    plot_line_size = 60 * 5  # 5분
    save_periods_minutes = 10


class Model:
    model_directory = str(ROOT_PATH / "artifacts")
    model_file_name = "pca_production.pkl"
