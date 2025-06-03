from PIL import Image

import os
import numpy as np
import pandas as pd

from torchvision import transforms

from weeds_detector.utils.padding import expand2square


def preprocess_features(X, output_folder):
    # X est le folder où se trouvent les images à traiter
    # Output folder est le folder où rajouter les images traitées

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
