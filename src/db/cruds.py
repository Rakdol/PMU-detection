import datetime
from logging import getLogger
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from src.db.models import PmuData
from src.utils import logger


def get_data_count(db: Session) -> int:
    return db.query(PmuData).count()


def select_pmu_from_btw_time(
    db: Session,
    start_time: str,
    end_time: str,
    pd_sql=True,
) -> Union[Select, List[PmuData]]:

    query = db.query(PmuData).filter(PmuData.timestamp.between(start_time, end_time))

    if pd_sql:
        pmu_data = query.all()
        # ORM 객체에서 데이터를 추출하여 리스트로 변환
        data = [
            {
                "timestamp": row.timestamp,
                "value": float(row.value),  # PmuData의 필드명에 맞게 수정
                "key": row.key,  # 필요한 필드 추가
            }
            for row in pmu_data
        ]

        # Pandas DataFrame으로 변환
        df = pd.DataFrame(data)
        try:
            pivot = df.pivot(index="timestamp", columns="key", values="value")
            return pivot
        except KeyError as e:
            logger.info(f" Data Base not retrived {e}")
            return

    return query.all()  # 리스트 반환 (ORM 객체)


# @log_execution_time
def select_pmu_by_key(db: Session, key: str, limit=500) -> List[PmuData]:
    if limit is not None:
        return db.query(PmuData).filter(PmuData.key == key).limit(limit).all()
    return db.query(PmuData).filter(PmuData.key == key).all()
