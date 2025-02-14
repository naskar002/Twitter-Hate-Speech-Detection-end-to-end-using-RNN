from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    imbalanced_data_file_path:str
    raw_data_file_path:str

@dataclass
class DataTransformationArtifacts:
    transformed_data_path:str

@dataclass
class ModelTrainerArtifacts:
    trained_model_path:str
    x_test_path:str
    y_test_path:str