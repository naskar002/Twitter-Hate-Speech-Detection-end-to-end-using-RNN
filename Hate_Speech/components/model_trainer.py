import os
import sys
import pickle
import pandas as pd
import numpy as np
from Hate_Speech.logger import logging
from Hate_Speech.contant import *
from Hate_Speech.exception import CustomException
from sklearn.model_selection import train_test_split
from Hate_Speech.ml.tokenizer import Tokenizer
from Hate_Speech.entity.config_entity import ModelTrainerConfig
from Hate_Speech.entity.artifact_entity import ModelTrainerArtifacts,DataTransformationArtifacts
from Hate_Speech.ml.models import ModelArchitecture
from Hate_Speech.contant import *
from torch.utils.data import TensorDataset,DataLoader

class ModelTrainer:
    def __init__(self,data_transformation_artifacts:DataTransformationArtifacts,model_trainer_config:ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config
        
    def spliting_data(self,csv_path):
        try:
            logging.info("Entered the spliting_data method")
            logging.info("Reading data")
            df = pd.read_csv(csv_path)
            logging.info("Splitting data into train and test")
            X = df[TWEET]
            y = df[LABEL]

            logging.info("Applying train test split")
            X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=RANDOM_STATE,test_size=0.25)
            logging.info("Exited spliting_data method")
            return X_train,X_test,y_train,y_test
        
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def tokenizing(self,X_train):
        try:
            logging.info("Entered tokenizing method of ModelTainer")
            tokenizer = Tokenizer(self.model_trainer_config.MAX_WORDS)
            X_train = X_train.fillna("").astype(str) 
            tokenizer.fit_on_texts(X_train)
            sequences = tokenizer.text_to_sequences(X_train)
            logging.info(f"Converting text to sequences {sequences}")
            sequence_tensor = tokenizer.pad_sequences(sequences=sequences,maxlen=self.model_trainer_config.MAX_LEN)
            logging.info(f"Sequence tensor is {sequence_tensor}")
            return sequence_tensor,tokenizer
        
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def creating_dataloader(self,sequence_tensor,target,test_data):
        # sequence_tensor: X_tarin or X_test
        # target: y_train or y_test
        try:
            logging.info("Entered creating_dataloader method")
            sequence_tensor = sequence_tensor.to(DEVICE)
            if isinstance(target, pd.Series):
                target = target.to_numpy()  # Convert Series to NumPy array
            elif not isinstance(target, (np.ndarray, list, torch.Tensor)):
                raise ValueError(f"Unexpected type for target: {type(target)}")
            target_tensor = torch.tensor(target,dtype=torch.float32).to(DEVICE)

            # creating the tensor dataset
            dataset = TensorDataset(sequence_tensor, target_tensor)
            if test_data:
                loader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE*2,shuffle=False,num_workers=0)
                logging.info("Dataloader created for training")
            else:
                loader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE, shuffle=True,num_workers=0)
                logging.info("Dataloader created for testing")
            return loader
        except Exception as e:
            raise CustomException(e,sys) from e

        
    def initiate_model_trainer(self)-> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            logging.info("Entered the initiate_model_trainer funcnction")
            X_train,X_test,y_train,y_test = self.spliting_data(self.data_transformation_artifacts.transformed_data_path)
            model_architecture = ModelArchitecture()

            model_trainer,model =  model_architecture.get_model()
            logging.info(f"Xtrain Size: {X_train.shape}")
            logging.info(f"Xtest Size: {X_test.shape}")

            sequence_tensor,tokenizer = self.tokenizing(X_train=X_train)

            X_test = X_test.fillna("").astype(str) 
            test_sequence = tokenizer.text_to_sequences(X_test)
            test_sequence_tensor = tokenizer.pad_sequences(sequences=test_sequence,maxlen=self.model_trainer_config.MAX_LEN)

            train_loader = self.creating_dataloader(sequence_tensor=sequence_tensor,target=y_train,test_data=False)

            val_loader = self.creating_dataloader(sequence_tensor=test_sequence_tensor,target=y_test,test_data=True)

            logging.info("Starting model training")
            model_trainer.fit(model,train_loader,val_loader)

            logging.info("Model training finished")

            with open ('tokenizer.pickle','wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR,exist_ok=True)


            logging.info("Saving the model")
            model_trainer.save_checkpoint(self.model_trainer_config.TRAINED_MODEL_PATH)
            X_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)

            X_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            model_trainer_artifacts = ModelTrainerArtifacts(trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH)
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e,sys) from e