import os
from datetime import datetime
import torch

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

# Data Transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = "DataTransformationArtifacts"
TRANSFORMED_FILE_NAME = "final.csv"
DATA_DIR = "data"
ID = "id"
AXIS = 1
INPLACE = True
DROP_COLUMNS = ['Unnamed: 0','count','hate_speech','offensive_language','neither']
CLASS = 'class'

# Model Training constants
MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"
TRAINED_MODEL_DIR = "trained_model"
TRAINED_MODEL_NAME = "model.ckpt"
X_TEST_FILE_NAME = "X_test.csv"
Y_TEST_FILE_NAME = "Y_test.csv"

X_TRAIN_FILE_NAME = 'X_train.csv'

RANDOM_STATE = 32
EPOCH = 5
BATCH_SIZE = 32

# Model Architecture constants
MAX_WORDS = 50000
MAX_LEN = 300
LEARNING_RATE = 1e-3
OPTIMIZER_NAME = "Adam"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
