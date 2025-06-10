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

    files_list = get_all_files_path_and_name_in_directory(f"croped_images/croped_{CROPED_SIZE}", extensions = [".png"])

    output_dir, folder_exist = create_folder(f'images_preprocessed/croped_images_resized_{CROPED_SIZE}/{RESIZED}x{RESIZED}')

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

    y = np.zeros(len(X_prepro))
    i = -1

    for file_path, file_name in files_list:
        i += 1
        if file_name[-5] == '1':
            y[i] = 1
    y = pd.Series(y)

    return X_prepro, y

def preprocess_single_image(img: Image.Image) -> np.ndarray:
    transform = transforms.PILToTensor()

    # Adapté de ta fonction expand2square + resize(128,128)
    new_image = expand2square(img, (0, 0, 0)).resize((128, 128))

    tensor = transform(new_image).permute(1, 2, 0).numpy()  # (H,W,C)
    tensor = tensor / 255.0
    tensor = np.expand_dims(tensor, axis=0)  # batch dimension

    return tensor
