import os

class GcloudSync:

    def sync_folder_to_gcloud(self,gcp_bucket_url,filepath,filename):
        command = f"gcloud storage cp {filepath}/{filename} gs://{gcp_bucket_url}"
        os.system(command)

    def sync_folder_from_gcloud(self, gcp_bucket_url, destination, filename):
        command = f"gcloud strorage cp gs://{gcp_bucket_url}/{filename} {destination}/{filename}"
        os.system(command)