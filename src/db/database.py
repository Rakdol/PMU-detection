import os
from contextlib import contextmanager
from logging import getLogger

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.db.configurations import PdcDBConfigurations

logger = getLogger(__name__)

logger.info(
    f"Source DB Configuration URL: {PdcDBConfigurations.sql_alchemy_database_url}"
)

pdc_engine = create_engine(
    PdcDBConfigurations.sql_alchemy_database_url,
    pool_recycle=3600,
    echo=False,
)

logger.info(f"Source DB connected: {pdc_engine}")

SessionPDC = sessionmaker(autocommit=False, autoflush=False, bind=pdc_engine)

BasePDC = declarative_base()
