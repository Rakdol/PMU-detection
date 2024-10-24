import os
import json
from pathlib import Path

path = str(Path(__file__).resolve().parents[1] / "secret.json")

with open(path, "r") as f:
    secret = json.load(f)


class PdcDBConfigurations:
    postgres_username = secret.get("POSTGRES_USER", "postgres")
    postgres_password = secret.get("POSTGRES_PASSWORD", None)
    postgres_port = int(secret.get("POSTGRES_PORT", 5432))
    postgres_db = secret.get("POSTGRES_DB", "openpdc")
    postgres_server = secret.get("POSTGRES_SERVER", "postgres-server")
    sql_alchemy_database_url = f"postgresql://{postgres_username}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"
