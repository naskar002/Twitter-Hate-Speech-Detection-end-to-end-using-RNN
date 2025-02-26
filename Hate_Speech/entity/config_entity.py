from dataclasses import dataclass
from Hate_Speech.contant import *
import os


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME = BUCKET_NAME
        self.ZIP_FILE_NAME = ZIP_FILE_NAME
        self.DATA_INGESTION_ARTIFACTS_DIR:str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_ARTIFACTS_DIR:str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATA_INGESTION_IMBALANCE_DATA_DIR) #STORES THE IMBALANCED DATA
        self.NEW_DATA_ARTIFACTS_DIR:str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATA_INGESTION_RAW_DATA_DIR)#stroe the raw data
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,ZIP_FILE_NAME) #gives the path of the zip file

@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR:str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRANSFORMED_FILE_PATH:str = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,TRANSFORMED_FILE_NAME)
        self.ID = ID
        self.AXIS = AXIS
        self.INPLACE = INPLACE
        self.DROP_COLUMNS = DROP_COLUMNS
        self.CLASS = CLASS
        self.LABEL = LABEL
        self.TWEET = TWEET

@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.TRAINED_MODEL_DIR:str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_MODEL_PATH:str = os.path.join(self.TRAINED_MODEL_DIR,TRAINED_MODEL_NAME)
        self.X_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR,X_TEST_FILE_NAME)
        self.Y_TEST_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR,Y_TEST_FILE_NAME)
        self.X_TRAIN_DATA_PATH = os.path.join(self.TRAINED_MODEL_DIR,X_TRAIN_FILE_NAME)
        self.MAX_WORDS = MAX_WORDS
        self.MAX_LEN = MAX_LEN
        self.LABEL = LABEL
        self.TWEET = TWEET
        self.RANDDOM_STATE = RANDOM_STATE
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCH

@dataclass
class ModelEvaluationConfig:  
    def __init__(self):
        self.MODEL_EVALUATION_MODDEL_DIR:str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_EVALUATION_ARTIFACTS_DIR)
        self.BEST_MODEL_DIR_PATH:str = os.path.join(self.MODEL_EVALUATION_MODDEL_DIR,BEST_MODEL_DIR)
        self.BUCKET_NAME = BUCKET_NAME
        self.MODEL_NAME = MODEL_NAME   

@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.TRAINED_MODEL_PATH = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR)
        self.BUCKET_NAME = BUCKET_NAME
        self.MODEL_NAME = MODEL_NAME