import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT_PATH))

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y')}.log"
logs_path = os.path.join(ROOT_PATH, "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 실행 시간 측정 데코레이터
def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 시작 시간 기록
        result = func(*args, **kwargs)
        end_time = time.time()  # 종료 시간 기록
        execution_time = end_time - start_time  # 실행 시간 계산
        logger.info(
            f"Function '{func.__name__}' executed in {execution_time:.4f} seconds"
        )
        return result

    return wrapper
