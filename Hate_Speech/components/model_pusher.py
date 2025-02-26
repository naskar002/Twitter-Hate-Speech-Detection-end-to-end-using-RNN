import sys
from Hate_Speech.logger import logging
from Hate_Speech.exception import CustomException
from Hate_Speech.configuration.gcloud_syncer import GcloudSync
from Hate_Speech.entity.config_entity import ModelPusherConfig
from Hate_Speech.entity.artifact_entity import ModelPusherArtifacts

class ModelPusher:
    def __init__(self, model_pusher_config:ModelPusherConfig):
        self.model_pusher_config = model_pusher_config
        self.gcloud = GcloudSync()

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        logging.info("Entered initiate_model_pusher method")
        try:
            self.gcloud.sync_folder_to_gcloud(gcp_bucket_url=self.model_pusher_config.BUCKET_NAME,
                                                filepath=self.model_pusher_config.TRAINED_MODEL_PATH,
                                                filename=self.model_pusher_config.MODEL_NAME)
            
            logging.info("Uploaded best model to gcloud storage")
            model_pusher_artifact  = ModelPusherArtifacts(bucket_name=self.model_pusher_config.BUCKET_NAME)
            logging.info("Exited the initiate model_pusher method of ModelTrainer class")
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e, sys) from e