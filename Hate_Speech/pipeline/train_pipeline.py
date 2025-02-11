import os
import sys
from Hate_Speech.logger import logging
from Hate_Speech.exception import CustomException
from Hate_Speech.components.data_ingestion import DataIngestion
from Hate_Speech.entity.config_entity import (DataIngestionConfig)
from Hate_Speech.entity.artifact_entity import (DataIngestionArtifacts)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered start_data_ingestion method of TrainPipeline")
        try:
            logging.info("Getting data from Gcloud")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)

            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train and valid from Gcloud Storage")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class.")
        except Exception as e:
            raise CustomException(e,sys) from e
        

    def run_pipeline(self):
        logging.info("Entered run_pipeline method of TrainPipeline")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()

            logging.info("Exited the run_pipeline method of TrainPipeline class")
        except Exception as e:
            raise CustomException(e,sys) from e