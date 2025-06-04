import os
import json
from weeds_detector.params import *
from google.cloud import storage
import requests

def get_filepath(filename: str):
    """
    Build filepath differently if files is saved locally or in GCP.
    """
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
    """
    Build filepath with file in folders differently if files is saved locally or in GCP.
    """
    path = os.path.join(*directories, filename)
    return get_filepath(path)

def get_json_content(filename):
    file = get_filepath(filename)

    if FILE_ORIGIN == 'gcp':
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        blob = bucket.blob(f"data/{filename}")
        str_json = blob.download_as_text()
        return json.loads(str_json)
    else:
        with open(file, "r") as f:
            return json.load(f)
