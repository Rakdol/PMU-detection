import os
import time
from datetime import datetime
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import psycopg2


def create_table(db_connect):
    # create_table_query = """
    # CREATE TABLE IF NOT EXISTS pmu_data (
    #     id SERIAL PRIMARY KEY,
    #     timestamp timestamp,
    #     frequency float8,
    #     rocof float8,
    #     voltage_r float8,
    #     voltage_r_angle float8,
    #     voltage_s float8,
    #     voltage_s_angle float8,
    #     voltage_t float8,
    #     voltage_t_angle float8,
    #     voltage_z float8,
    #     voltage_z_angle float8,
    #     current_1 float8,
    #     current_1_angle float8,
    #     current_2 float8,
    #     current_2_angle float8,
    #     current_3 float8,
    #     current_3_angle float8,
    #     current_4 float8,
    #     current_4_angle float8,
    #     current_5 float8,
    #     current_5_angle float8,
    #     current_6 float8,
    #     current_6_angle float8,
    #     current_7 float8,
    #     current_7_angle float8,
    #     current_8 float8,
    #     current_8_angle float8
    # );
    # """

    create_table_query = """
    CREATE TABLE IF NOT EXISTS pmu_data (
        signalid SERIAL PRIMARY KEY,
        key VARCHAR(30),
        timestamp VARCHAR(30),
        value VARCHAR(30)
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
    # insert_row_query = """
    # INSERT INTO pmu_data (
    #     timestamp, frequency, rocof, voltage_r, voltage_r_angle, voltage_s, voltage_s_angle, voltage_t, voltage_t_angle, voltage_z, voltage_z_angle, current_1, current_1_angle, current_2, current_2_angle, current_3, current_3_angle, current_4, current_4_angle, current_5, current_5_angle, current_6, current_6_angle, current_7, current_7_angle, current_8, current_8_angle
    # ) VALUES (
    #     NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    # );
    # """

    # Get the current timestamp
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Define the SQL query for insertion
    insert_query = """
        INSERT INTO pmu_data (timestamp, key, value) 
        VALUES (%s, %s, %s);
    """

    with db_connect.cursor() as cur:
        # Loop through each row and column in the DataFrame to insert into the database
        for col in data.columns:
            for index, value in data[col].items():
                cur.execute(insert_query, (current_timestamp, col, value))

        # Commit the transaction
        db_connect.commit()

    # # None (NaN) 값은 자동으로 NULL로 처리됨
    # data_tuple = (
    #     None if np.isnan(data["Frequency"]) else float(data["Frequency"]),
    #     None if np.isnan(data["DeFrequency"]) else float(data["DeFrequency"]),
    #     None if np.isnan(data["Voltage_R"]) else float(data["Voltage_R"]),
    #     None if np.isnan(data["Voltage_R_Angle"]) else float(data["Voltage_R_Angle"]),
    #     None if np.isnan(data["Voltage_S"]) else float(data["Voltage_S"]),
    #     None if np.isnan(data["Voltage_S_Angle"]) else float(data["Voltage_S_Angle"]),
    #     None if np.isnan(data["Voltage_T"]) else float(data["Voltage_T"]),
    #     None if np.isnan(data["Voltage_T_Angle"]) else float(data["Voltage_T_Angle"]),
    #     None if np.isnan(data["Voltage_Z"]) else float(data["Voltage_Z"]),
    #     None if np.isnan(data["Voltage_Z_Angle"]) else float(data["Voltage_Z_Angle"]),
    #     None if np.isnan(data["Current_I1"]) else float(data["Current_I1"]),
    #     None if np.isnan(data["Current_I1_Angle"]) else float(data["Current_I1_Angle"]),
    #     None if np.isnan(data["Current_I2"]) else float(data["Current_I2"]),
    #     None if np.isnan(data["Current_I2_Angle"]) else float(data["Current_I2_Angle"]),
    #     None if np.isnan(data["Current_I3"]) else float(data["Current_I3"]),
    #     None if np.isnan(data["Current_I3_Angle"]) else float(data["Current_I3_Angle"]),
    #     None if np.isnan(data["Current_I4"]) else float(data["Current_I4"]),
    #     None if np.isnan(data["Current_I4_Angle"]) else float(data["Current_I4_Angle"]),
    #     None if np.isnan(data["Current_I5"]) else float(data["Current_I5"]),
    #     None if np.isnan(data["Current_I5_Angle"]) else float(data["Current_I5_Angle"]),
    #     None if np.isnan(data["Current_I6"]) else float(data["Current_I6"]),
    #     None if np.isnan(data["Current_I6_Angle"]) else float(data["Current_I6_Angle"]),
    #     None if np.isnan(data["Current_I7"]) else float(data["Current_I7"]),
    #     None if np.isnan(data["Current_I7_Angle"]) else float(data["Current_I7_Angle"]),
    #     None if np.isnan(data["Current_I8"]) else float(data["Current_I8"]),
    #     None if np.isnan(data["Current_I8_Angle"]) else float(data["Current_I8_Angle"]),
    # )

    # print(insert_row_query)
    # with db_connect.cursor() as cur:
    #     cur.execute(insert_row_query, data_tuple)
    #     db_connect.commit()


def generate_data(db_connect, df):
    while True:
        insert_data(db_connect, df.sample(1).drop("timestamp", axis=1))
        time.sleep(1 / 60)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--db-host", dest="db_host", type=str, default="localhost")
    parser.add_argument(
        "--csv-path", dest="csv_path", type=str, default="/usr/app/pmu.csv"
    )
    # parser.add_argument("--drop", dest="drop", type=bool, default=False)

    args = parser.parse_args()

    print("DataBase Host", args.db_host)
    print("Data File Path", args.csv_path)

    db_connect = psycopg2.connect(
        user="admin",
        password="1234",
        host=args.db_host,
        port=5432,
        database="pmudb",
    )

    create_table(db_connect)
    df = pd.read_csv(args.csv_path)
    generate_data(db_connect, df)
