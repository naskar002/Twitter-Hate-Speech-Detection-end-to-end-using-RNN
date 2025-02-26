import sys
import os
import torch
import pickle
import numpy as np
import pandas as pd
from Hate_Speech.logger import logging
from Hate_Speech.exception import CustomException
from torch.nn.utils.rnn import pad_sequence
from Hate_Speech.contant import *
from Hate_Speech.configuration.gcloud_syncer import GcloudSync
from Hate_Speech.entity.config_entity import ModelEvaluationConfig
from Hate_Speech.entity.artifact_entity import ModelEvaluationArtifacts,ModelTrainerArtifacts,DataTransformationArtifacts
import pytorch_lightning as pl
from Hate_Speech.ml.models import LSTMModel
from torch.utils.data import DataLoader,TensorDataset


class ModelEvaluation:
    def __init__(self,model_evaluation_config:ModelEvaluationConfig,model_trainer_artifacts:ModelTrainerArtifacts,data_transformation_artifacts:DataTransformationArtifacts):

        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.gcloud = GcloudSync()


    def get_best_model_from_gcloud(self) -> str:
        """
        :return: Fetch best model from gcloud storage and store inside best model directory path
        """
        try:
            logging.info("Fetching best model from Gcloud")
            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH,exist_ok=True)
            self.gcloud.sync_folder_from_gcloud(self.model_evaluation_config.BUCKET_NAME,self.model_evaluation_config.BEST_MODEL_DIR_PATH,self.model_evaluation_config.MODEL_NAME)

            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,self.model_evaluation_config.MODEL_NAME)

            logging.info("Exited the get_best_model_from_gcloud method")
            return best_model_path
            
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def evaluate(self,model):
        """

        :param model: Currently trained model or best model from gcloud storage
        :param data_loader: Data loader for validation dataset
        :return: loss
        """
        try:
            logging.info("Entered evaluate method of ModelEvaluation class")
            print(self.model_trainer_artifacts.x_test_path)

            X_test = pd.read_csv(self.model_trainer_artifacts.x_test_path,index_col=0)
            y_test  = pd.read_csv(self.model_trainer_artifacts.y_test_path,index_col=0)

            with open ('tokenizer.pickle','rb') as handle:
                tokenizer = pickle.load(handle)


            X_test = X_test[TWEET].fillna("").astype(str)
            test_sequence = tokenizer.text_to_sequences(X_test)
            test_sequence_tesnsor = tokenizer.pad_sequences(test_sequence,MAX_LEN)

            tensor_dataset = TensorDataset(test_sequence_tesnsor)
            test_loader = DataLoader(dataset=tensor_dataset,batch_size=32, shuffle=False)

            y_hats = []
            sig_fn = torch.nn.Sigmoid()

            for batch in test_loader:
                batch = batch[0]
                model.eval()
                with torch.no_grad():
                    y_hats.append(model(batch))

            y_hat = torch.cat(y_hats)
            y_pred_class = (sig_fn(y_hat) > 0.5).float()
            
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)
            accuracy = (y_pred_class == y_test_tensor).float().mean().item()

            logging.info(f"The test accuracy is {accuracy}")
            return accuracy
        except Exception as e:
            raise CustomException(e,sys) from e
        

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        logging.info("Initiate Model Evaluation...")
        try:
            logging.info("Loading currently trained model")
            trained_model = LSTMModel.load_from_checkpoint(self.model_trainer_artifacts.trained_model_path)
            with open('tokenizer.pickle','rb') as handle:
                load_tokenizer = pickle.load(handle)

            trained_model_accuracy = self.evaluate(trained_model)

            logging.info("Fetch best model from gcloud")
            best_model_path = self.get_best_model_from_gcloud()

            logging.info("Is best model present in gcloud?")
            if os.path.isfile(best_model_path) is False:
                is_model_accepted = True
                logging.info("gcloud storage model is false and currently trained model is accepted")
            else:
                logging.info("Load best model fetch from gcloud")
                best_model = LSTMModel.load_from_checkpoint(best_model_path)
                best_model_accuracy = self.evaluate(best_model)

                logging.info("Comparing loss between best_model_loss and trained_model_loss ? ")
                if best_model_accuracy>trained_model_accuracy:
                    is_model_accepted = False
                    logging.info("gcloud storage model loss is better and currently trained model is rejected")
                else:
                    is_model_accepted = True
                    logging.info("gcloud storage model loss is worse and currently trained model is accepted")

            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("Returning ModelEvaluationArtifacts")
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e,sys) from e