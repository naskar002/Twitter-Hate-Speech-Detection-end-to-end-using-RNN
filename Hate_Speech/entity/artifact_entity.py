from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    imbalanced_data_file_path:str
    raw_data_file_path:str