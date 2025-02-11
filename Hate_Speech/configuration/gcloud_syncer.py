import os

class GcloudSync:

    def sync_folder_to_gcloud(self,gcp_bucket_url,filepath,filename):
        command = f"gsutil cp {filepath}/{filename} gs://{gcp_bucket_url}"
        os.system(command)

    def sync_folder_from_gcloud(self, gcp_bucket_url, destination, filename):
        command = f"gsutil cp gs://{gcp_bucket_url}/{filename} {destination}/{filename}"
        os.system(command)