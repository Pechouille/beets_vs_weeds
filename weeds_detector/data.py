import os
from weeds_detector.params import *
from google.cloud import storage

def get_filepath(filename: str):
    if FILE_ORIGIN == 'local':
        filepath = os.path.join(LOCAL_DATA_PATH, filename)
        if not os.path.exists(filepath):
            print(f"❌ File not found : {filename}")
            return None

        return filepath

    if FILE_ORIGIN == 'gcp':
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        try:
            filepath = bucket.get_blob(f"data/{filename}").public_url
            return filepath
        except AttributeError:
            print(f"❌ File not found : {filename}")
            return None

def get_filepath_in_directories(filename: str, directories: list):
    directories.append(filename)
    path = '/'.join(directories)
    return get_filepath(path)
