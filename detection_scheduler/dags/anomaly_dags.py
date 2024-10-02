import datetime
import requests
import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.sql_sensor import SqlSensor

from sqlalchemy.orm import Session


default_args = {
    "owner": "kWater",
    "depends_on_past": False,
    "start_date": datetime.datetime(2024, 10, 2),
    "retries": 2,
    "retry_delay": datetime.timedelta(seconds=10),
}

with DAG(
    "PMU_Event_Detector",
    description="PMU Detector",
    default_args=default_args,
    schedule_interval=datetime.timedelta(seconds=1),  # Schedule Trigger
    catchup=False,
) as dag: