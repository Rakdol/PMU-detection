from sqlalchemy import (
    Column,
    String,
)
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.types import JSON
from src.db.database import BasePDC


class PmuData(BasePDC):

    __tablename__ = "pmu_data"

    signalid = Column(String(50), nullable=True, primary_key=True)
    key = Column(String(20), nullable=True)
    timestamp = Column(String(30), nullable=True)
    value = Column(String(50), nullable=True)

    # Indicate that the table has no primary key
    __table_args__ = {"extend_existing": True}
