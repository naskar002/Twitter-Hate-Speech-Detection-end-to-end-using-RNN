import os
import io
import sys
import torch
import torch.nn as nn
import pickle
from Hate_Speech.logger import logging
from Hate_Speech.contant import *
from Hate_Speech.exception import CustomException
from Hate_Speech.configuration.gcloud_syncer import GcloudSync
from Hate_Speech.components.data_transformation import DataTransformation
from Hate_Speech.entity.config_entity import DataTransformationConfig
from Hate_Speech.entity.artifact_entity import DataIngestionArtifacts
from Hate_Speech.ml.models import LSTMModel


class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts","PredictModel")
        self.gcloud = GcloudSync()
        self.data_transformation =  DataTransformation(data_transformation_config=DataTransformationConfig,data_ingestion_artifacts=DataIngestionArtifacts)
        

    def get_model_from_gcloud(self) -> str:
        """
        Method Name :   get_model_from_gcloud
        Description :   This method to get best model from google cloud storage
        Output      :   best_model_path
        """
        logging.info("Entered get_model_from_gcloud method of prediction pipeline")
        try:
            os.makedirs(self.model_path,exist_ok= True)
            self.gcloud.sync_folder_from_gcloud(self.bucket_name,self.model_path,self.model_name)
            best_model_path = os.path.join(self.model_path,self.model_name)
            logging.info("Exited the get_model_from_gcloud method in prediction pipeline")
            return best_model_path

        except Exception as e:
            raise CustomException(e,sys) from e
        
    
    def predict(self,best_model_path,text):
        logging.info("Running the predict method")
        try:
            # best_model_path:str = self.get_model_from_gcloud()
            load_model = LSTMModel.load_from_checkpoint(best_model_path)

            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)

            text = self.data_transformation.concat_data_cleaning(text)
            text = [text]
            seq = load_tokenizer.text_to_sequences(text)
            padded = load_tokenizer.pad_sequences(seq,MAX_LEN)
            sigmoid = nn.Sigmoid()
            pred = sigmoid(load_model(padded)).item()
            if pred > 0.5:
                print("Hate or Abusive")
                return "Hate or Abusive"
            else:
                print("Not Hate or Abusive")
                return "Not Hate or Abusive"
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def run_pipeline(self,text):
        logging.info("Running the run_pipeline method")
        try:
            model_path = os.path.join("artifacts","PredictModel",self.model_name)
            best_model_path = model_path
            if not os.path.isfile(model_path):
                best_model_path = self.get_model_from_gcloud()
            predicted_text = self.predict(best_model_path,text)
            logging.info("Exited the run_pipeline method of prediction class")
            return predicted_text
        except Exception as e:
            raise CustomException(e,sys) from e