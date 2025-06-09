from PIL import Image

import os
import numpy as np
import pandas as pd
from io import BytesIO
import requests


from torchvision import transforms

from weeds_detector.utils.padding import expand2square
from weeds_detector.data import get_all_files_path_and_name_in_directory
from weeds_detector.params import *

from weeds_detector.ml_logic.preprocess_model_segm_class import create_folder
from weeds_detector.utils.images import save_image


def preprocess_features():
    """
    Output folder = empty folder needed to add the preprocessed images
    """

    list_of_tensors = []
    transform = transforms.Compose([transforms.PILToTensor()])

    files_list = get_all_files_path_and_name_in_directory(f"croped_images/croped_{CROPED_SIZE}", extensions = [".png"])

    for file_path, file_name in files_list:

        response = requests.get(file_path)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        resized_value = int(RESIZED)
        new_image = expand2square(img, (0, 0, 0)).resize((resized_value,resized_value))

        output_folder = create_folder('images_preprocessed_model1')
        save_image(new_image, output_folder, f"preprocessed_{file_name}")

        transf = transform(new_image)
        tensor = transf.permute(1, 2, 0)
        list_of_tensors.append(tensor)

    X_prepro = np.array([tensor.numpy() for tensor in list_of_tensors])
    X_prepro = X_prepro / 255

    y = np.zeros(len(X_prepro))
    i = -1
    for image_name in os.listdir(X):
        i += 1
        if image_name[-5] == '1':
            y[i] = 1
    y = pd.Series(y)

    return X_prepro, y
