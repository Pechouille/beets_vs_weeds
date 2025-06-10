from PIL import Image

import os
import numpy as np
import pandas as pd
from io import BytesIO
import requests

from tensorflow.keras.utils import img_to_array
from weeds_detector.utils.padding import expand2square
from weeds_detector.data import get_all_files_path_and_name_in_directory
from weeds_detector.params import *
from google.cloud import storage

from weeds_detector.ml_logic.preprocess_model_segm_class import create_folder, transform_image
from weeds_detector.utils.images import save_image


def preprocess_features():
    list_of_tensors = []
    files_list = get_all_files_path_and_name_in_directory(f"croped_images_UNET", extensions = [".png"])
    output_dir, folder_exist = create_folder(f'images_preprocessed_UNET/{RESIZED}x{RESIZED}')
    storage_client = storage.Client()
    source_bucket = storage_client.bucket(BUCKET_NAME)

    for file_path, file_name in files_list:
        print(f"Get image : preprocessed_{file_name} in bucket {output_dir}")
        source_blob = source_bucket.blob(os.path.join(output_dir, f"preprocessed_{file_name}"))
        image_path = source_blob.public_url
        response = requests.get(image_path)

        if response.status_code == 200:
            print(f"Response code : {response.status_code}")
            new_image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            print(f"Response code | {response.status_code} : Image not found transform image")
            new_image = transform_image(file_name, file_path, output_dir)

        image = img_to_array(new_image)
        image = image / 255.0
        list_of_tensors.append(image)

    X_prepro = np.stack(list_of_tensors, axis=0)

    return X_prepro
