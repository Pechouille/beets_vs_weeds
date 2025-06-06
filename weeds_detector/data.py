import os
import json
import logger
from weeds_detector.params import *
from google.cloud import storage
from typing import Set

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


def get_folderpath(foldername: str):
    """
    Build filepath differently if files are saved locally or in GCP.
    """
    if FILE_ORIGIN == 'local':
        folderpath = os.path.join(LOCAL_DATA_PATH, foldername)
        if not os.path.exists(folderpath):
            print(f"❌ Folder not found : {foldername}")
            return None
        return folderpath

    elif FILE_ORIGIN == 'gcp':
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        folder_blob_name = foldername if foldername.endswith('/') else foldername + '/'
        blob = bucket.blob(folder_blob_name)

        if blob.exists():
            return f"gs://{BUCKET_NAME}/{folder_blob_name}"
        else:
            print(f"❌ Folder not found : {foldername}")
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

def get_existing_files(dir: str) -> Set[str]:
    """Get set of already processed crop filenames for skip logic"""
    existing_files = set()

    if FILE_ORIGIN == 'local':
        if os.path.exists(dir):
            for filename in os.listdir(dir):
                if filename.endswith('.png'):
                    existing_files.add(filename)

    elif FILE_ORIGIN == 'gcp':
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            blob_prefix = f"data/{dir}/"

            for blob in bucket.list_blobs(match_glob="**.png", prefix=blob_prefix):
                existing_files.add(os.path.basename(blob.name))
        except Exception as e:
            logger.warning(f"Could not list existing files from GCP: {e}")

    return existing_files
