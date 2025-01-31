import sys
from Hate_Speech.configuration.gcloud_syncer import GcloudSync

gcl_sync = GcloudSync()
gcl_sync.sync_folder_from_gcloud("twitter-hate-speech-31012025","downlaod/dataset.zip","dataset.zip")