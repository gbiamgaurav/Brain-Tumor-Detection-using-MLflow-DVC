from brain_tumor import logger
from brain_tumor.pipeline.stage_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"Stage: {STAGE_NAME} Started!")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"Stage: {STAGE_NAME} Completed!")
except Exception as e:
    logger.exception(e)
    raise e 