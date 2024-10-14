import datetime
from logging import getLogger
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from src.db.models import PmuData
from src.utils import log_execution_time


logger = getLogger(__name__)


@log_execution_time
def get_data_count(db: Session) -> int:
    return db.query(PmuData).count()


@log_execution_time
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

        return df.pivot(index="timestamp", columns="key", values="value")

    return query.all()  # 리스트 반환 (ORM 객체)


@log_execution_time
def select_pmu_by_key(db: Session, key: str, limit=500) -> List[PmuData]:
    if limit is not None:
        return db.query(PmuData).filter(PmuData.key == key).limit(limit).all()
    return db.query(PmuData).filter(PmuData.key == key).all()


# if __name__ == "__main__":
#     session = Session_PDC()
#     data = session.query(PmuData).first()
#     start_time = datetime.now() - timedelta(hours=9)
#     end_time = start_time - timedelta(seconds=1)

#     data = select_pmu_from_btw_time(
#         db=Session_PDC(),
#         start_time=end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
#         end_time=start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
#     )

#     for d in data:
#         print(d.key, ": ", d.value)
