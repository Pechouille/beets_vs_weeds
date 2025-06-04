import os
import json
from weeds_detector.params import *
from google.cloud import storage

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
    with open(file, "r") as f:
        return json.load(f)

def extension_accepted(filename: str, extensions: list):
    extension = os.path.splitext(filename)[-1].lower()
    return extension in extensions

def get_all_files_path_and_name_in_directory(directory_path: str, extensions: list=[]):
    """
    Build filepath differently if files is saved locally or in GCP.
    """
    files_list = []
    if FILE_ORIGIN == 'local':
        directory_path = os.path.join(LOCAL_DATA_PATH, directory_path)
        if not os.path.exists(directory_path):
            print(f"❌ Directory not found : {directory_path}")
            return None


        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                if extensions != [] and extension_accepted(filename, extensions):
                    files_list.append([file_path, filename])
                elif extensions == []:
                    files_list.append([file_path, filename])

    if FILE_ORIGIN == 'gcp':
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        prefix = f"data/{directory_path}/"
        if extensions != []:
            blobs = []
            for extension in extensions:
                pattern = f"**{extension}"
                blobs += bucket.list_blobs(match_glob=pattern, prefix=prefix)
        else:
            blobs = bucket.list_blobs(prefix=prefix)

        if len(blobs) != 0:
            for blob in blobs:
                files_list.append([blob.public_url, blob.name.replace(prefix, "")])
        else:
            print(f"❌ Directory not found or zero files in this directory : {directory_path}")

    return files_list
