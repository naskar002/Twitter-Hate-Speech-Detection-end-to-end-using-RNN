import os
import sys
from zipfile import ZipFile
from Hate_Speech.logger import logging
from Hate_Speech.exception import CustomException
from Hate_Speech.configuration.gcloud_syncer import GcloudSync
from Hate_Speech.entity.config_entity import DataIngestionConfig
from Hate_Speech.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.gcl_sync = GcloudSync()

    def get_data_from_gcloud(self)->None:
        try:
            logging.info("Fetching data from Gcloud")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            self.gcl_sync.sync_folder_from_gcloud(self.data_ingestion_config.BUCKET_NAME,self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,self.data_ingestion_config.ZIP_FILE_NAME)

            logging.info("Extracted data successfully")

        except Exception as e:
            raise CustomException(e,sys) from e
        
    def unzip_and_clean(self):
        logging.info("Unzipping data...")
        try:
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH,'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)

                logging.info("Data unzipped successfully")
                return self.data_ingestion_config.DATA_ARTIFACTS_DIR,self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR #(imbalanced_Dataset,raw_dataset)
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Data ingestion started")
        
        try:
            self.get_data_from_gcloud()
            logging.info("Fetched Data from Gcloud Bucket")
            imbalance_data_file_path,raw_data_file_path = self.unzip_and_clean()
            logging.info("Unzipped and cleaned data")

            data_ingestion_artifacts = DataIngestionArtifacts(imbalanced_data_file_path=imbalance_data_file_path,raw_data_file_path=raw_data_file_path)

            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")

            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts
        
        except Exception as e:
            raise CustomException(e,sys) from e