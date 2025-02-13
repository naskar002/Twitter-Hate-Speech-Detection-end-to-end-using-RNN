import os
import sys
from Hate_Speech.logger import logging
from Hate_Speech.exception import CustomException
from Hate_Speech.components.data_ingestion import DataIngestion
from Hate_Speech.components.data_transformation import DataTransformation
from Hate_Speech.entity.config_entity import (DataIngestionConfig,DataTransformationConfig)
from Hate_Speech.entity.artifact_entity import (DataIngestionArtifacts,DataTransformationArtifacts)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered start_data_ingestion method of TrainPipeline")
        try:
            logging.info("Getting data from Gcloud")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)

            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train and valid from Gcloud Storage")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class.")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def start_data_transformation(self,data_ingestion_artifacts = DataIngestionArtifacts) -> DataTransformationArtifacts:

        logging.info("Entered start_data_transformation method of TrainPipeline")
        try:
            data_transformation = DataTransformation(data_ingestion_artifacts=data_ingestion_artifacts,data_transformation_config=self.data_transformation_config)

            data_transformation_artifacts = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed successfully")
            logging.info("Exited the start_data_transformation method of TrainPipeline class.")

            return data_transformation_artifacts
        
        except Exception as e:
            raise CustomException(e,sys) from e
        

    def run_pipeline(self):
        logging.info("Entered run_pipeline method of TrainPipeline")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()

            data_transformation_artifacts = self.start_data_transformation(data_ingestion_artifacts)

            logging.info("Exited the run_pipeline method of TrainPipeline class")
        except Exception as e:
            raise CustomException(e,sys) from e