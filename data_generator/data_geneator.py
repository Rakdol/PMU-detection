import os
import time
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import psycopg2

from pmu import VirtualPMU


def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS pmu_data (
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        frequency float8,
        voltage float8,
        current float8,
        phase_angle float8,
        label_name VARCHAR(30),
        label int
    );
    """

    print(create_table_query)

    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()


def drop_table(db_connect):
    drop_table_query = """
    DROP TABLE IF EXISTS pmu_data;
    """

    with db_connect.cursor() as cur:
        cur.execute(drop_table_query)
        db_connect.commit()


def insert_data(db_connect, data):
    # 파라미터화된 쿼리 사용
    insert_row_query = """
    INSERT INTO pmu_data (
        timestamp, frequency, voltage, current, phase_angle, label_name, label
    ) VALUES (
        NOW(), %s, %s, %s, %s, %s, %s
    );
    """

    # None (NaN) 값은 자동으로 NULL로 처리됨
    data_tuple = (
        (
            None if np.isnan(data.frequency) else float(data.frequency)
        ),  # NaN을 None으로 변환
        None if np.isnan(data.voltage) else float(data.voltage),
        None if np.isnan(data.current) else float(data.current),
        None if np.isnan(data.phase_angle) else float(data.phase_angle),
        str(data.label_name),  # 문자열 처리
        (
            int(data.label) if isinstance(data.label, (np.integer, int)) else data.label
        ),  # NumPy int -> Python int
    )

    # print(insert_row_query)
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query, data_tuple)
        db_connect.commit()


def generate_data(db_connect):
    PMU = VirtualPMU(sample_rate=60, event_rate=0.03, error_rate=0.03)
    while True:
        data = PMU.create_dataframe(duration=1)
        for i in range(data.shape[0]):
            insert_data(db_connect=db_connect, data=data.iloc[i, :])

        time.sleep(1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--db-host", dest="db_host", type=str, default="localhost")
    # parser.add_argument("--drop", dest="drop", type=bool, default=False)

    args = parser.parse_args()

    print("DataBase Host: ", args.db_host)
    # print("is Drop Table: ", args.drop)

    db_connect = psycopg2.connect(
        user="admin",
        password="1234",
        host=args.db_host,
        port=5432,
        database="machinedb",
    )

    create_table(db_connect)
    generate_data(db_connect)
