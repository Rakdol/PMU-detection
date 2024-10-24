import json
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]

path = str(Path(__file__).resolve().parents[0] / "secret.json")
with open(path, "r") as f:
    secret = json.load(f)


class Monitoring:
    update_step_seconds = 1
    buffer_size = 60 * 60 * 60  # 1초 60 샘플 - 1시간
    KST = 9
    plot_line_size = 60 * 5  # 5분
    save_periods_minutes = 10


class Model:
    model_directory = str(ROOT_PATH / "artifacts")
    model_file_name = "pca_production.pkl"


class BucketServer:
    bucket_server = secret.get("MINIO_SERVER", "localhost")
    bucket_port = int(secret.get("MINIO_PORT", 9000))
    bucket_access = secret.get("MINIO_ACCESS", None)
    bucket_secret = secret.get("MINIO_SECRET", None)
    bucket_name = "test"
