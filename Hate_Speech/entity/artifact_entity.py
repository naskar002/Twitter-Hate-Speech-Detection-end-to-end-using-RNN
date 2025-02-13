from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    imbalanced_data_file_path:str
    raw_data_file_path:str

@dataclass
class DataTransformationArtifacts:
    transformed_data_path:str