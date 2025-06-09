from PIL import Image

import os
import numpy as np
import pandas as pd

from torchvision import transforms

from weeds_detector.utils.padding import expand2square


def preprocess_features(X, output_folder):
    """
    X being the folder where the images to process are located
    Output folder is the empty folder needed to add the preprocessed images
    """

    list_of_tensors = []
    transform = transforms.Compose([transforms.PILToTensor()])

    for image_name in os.listdir(X):

        image_path = os.path.join(X, image_name)
        img = Image.open(image_path).convert("RGB")

        new_image = expand2square(img, (0, 0, 0)).resize((128,128))
        save_path = os.path.join(output_folder, image_name)

        new_image.save(save_path)

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

def preprocess_single_image(img: Image.Image) -> np.ndarray:
    transform = transforms.PILToTensor()

    # Adapté de ta fonction expand2square + resize(128,128)
    new_image = expand2square(img, (0, 0, 0)).resize((128, 128))

    tensor = transform(new_image).permute(1, 2, 0).numpy()  # (H,W,C)
    tensor = tensor / 255.0
    tensor = np.expand_dims(tensor, axis=0)  # batch dimension

    return tensor
