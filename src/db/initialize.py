from logging import getLogger

from src.db.database import BasePDC

logger = getLogger(__name__)


def create_pmu_tables(engine, checkfirst: bool = True):
    logger.info("Initialize tables if not exist.")
    BasePDC.metadata.create_all(engine, checkfirst=checkfirst)


def initialize_pmu_table(engine, checkfirst: bool = True):
    logger.info("Initialize tables")
    create_pmu_tables(engine=engine, checkfirst=checkfirst)
