import os
import re
import sys
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from Hate_Speech.logger import logging
from Hate_Speech.exception import CustomException
from Hate_Speech.entity.config_entity import DataTransformationConfig
from Hate_Speech.entity.artifact_entity import DataTransformationArtifacts,DataIngestionArtifacts

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,data_ingestion_artifacts:DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def imbalance_data_cleaning(self):

        try:
            logging.info("Entered imbalance_data_cleaning method of DataTransformation")
            imbalance_data = pd.read_csv(self.data_ingestion_artifacts.imbalanced_data_file_path)
            imbalance_data.drop(self.data_transformation_config.ID,axis=self.data_transformation_config.AXIS,inplace=self.data_transformation_config.INPLACE)
            logging.info(f"Exited imbalance_data_cleaning method of DataTransformation and returned {imbalance_data}")
            return imbalance_data
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def raw_data_cleaning(self):

        try:
            logging.info("Entered raw_data_cleaning_method of Data Transformation class")
            raw_data = pd.read_csv(self.data_ingestion_artifacts.raw_data_file_path)

            # dropped the columns 
            raw_data.drop(self.data_transformation_config.DROP_COLUMNS,axis=self.data_transformation_config.AXIS,inplace=self.data_transformation_config.INPLACE)

            # convert all 0's to 1's
            raw_data[self.data_transformation_config.CLASS].replace({0:1},inplace = self.data_transformation_config.INPLACE)

            # renamed columns from class to label
            raw_data.rename(columns = {self.data_transformation_config.CLASS:self.data_transformation_config.LABEL},inplace= True)

            # convert all 2's to 0's
            raw_data[self.data_transformation_config.LABEL].replace({2:0},inplace = True)

            logging.info(f"Exited raw_data_cleaning_method of DataTransformation and returned {raw_data}")
            return raw_data
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def concat_dataframe(self):

        try:
            logging.info("Entered concat_dataframe method of DataTransformation class")
            frame = [self.raw_data_cleaning(),self.imbalance_data_cleaning()]
            df = pd.concat(frame)
            logging.info(f"Exited concat_dataframe method of DataTransformation and returned {df}")
            return df
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def concat_data_cleaning(self,words):

        try:
            logging.info("Entered concat_data_cleaning method of DataTransformation class")
            stemmer = PorterStemmer()
            stop_words = set(stopwords.words('english'))
            words = str(words).lower()
            words = re.sub('\[.*?\]','',words)
            words = re.sub('https?://\S+|www\.\S+', '', words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)

            # The line below was causing the issue. Changed to check 'word' in stop_words
            words = [word for word in words.split(' ') if word not in stop_words]
            words=" ".join(words)
            words = [stemmer.stem(word) for word in words.split(' ')]
            words=" ".join(words)
            logging.info("Exited concat_data_cleaning method of DataTransformation")
            return words
        
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def initiate_data_transformation(self):

        try:
            logging.info("Data transformation started")
            self.imbalance_data_cleaning()
            self.raw_data_cleaning()
            df = self.concat_dataframe()
            df[self.data_transformation_config.TWEET] = df[self.data_transformation_config.TWEET].apply(self.concat_data_cleaning)

            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR,exist_ok=True)

            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH,index=False,header=True)

            data_transformation_artifact = DataTransformationArtifacts(transformed_data_path=self.data_transformation_config.TRANSFORMED_FILE_PATH)

            logging.info("returning the DataTransformationArtifacrs")
            return data_transformation_artifact
        
        except Exception as e:
            raise CustomException(e,sys) from e
    