import os
from datetime import datetime

#common constants
TIMESTAMP:str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR:str = os.path.join("artifacts",TIMESTAMP)
BUCKET_NAME = "twitter-hate-speech-31012025"
ZIP_FILE_NAME = "dataset.zip"
LABEL = "label"
TWEET = "tweet"

#Data ingestion constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATA_INGESTION_IMBALANCE_DATA_DIR = "imbalanced_data.csv"
DATA_INGESTION_RAW_DATA_DIR  = "raw_data.csv"