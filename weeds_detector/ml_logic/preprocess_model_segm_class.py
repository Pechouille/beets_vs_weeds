import pandas as pd
import numpy as np
import json
import os
import shutil
from torchvision import transforms
from PIL import Image

from weeds_detector.utils.padding import expand2square


csv = pd.read_csv('data/csv/image_characteristics.csv')

images_plus_de_10 = csv[csv.number_items_per_picture > 10][['id', 'filename', 'number_items_per_picture']]

json_path = '/Users/ramoisiaux/code/Pechouille/beets_vs_weeds/data/test:val:train/json_train_set.json'

with open(json_path, 'r') as f:
        data = json.load(f)

excluded_filenames = set(images_plus_de_10['filename'])
annotated_ids = set(img["image_id"] for img in data["annotations"])
excluded_id = set(images_plus_de_10['id'])


def preprocess_images(input_folder, output_folder, prepro_folder):
    """
    input folder being the folder where all images are located
    output folder is the empty folder that will contain only the images we need to preprocess
    prepro folder is the empty folder that will contain the images preprocessed
    """

    file_names = set(img["file_name"] for img in data.get("images", []) if img["file_name"] not in excluded_filenames and img["id"] in annotated_ids)

    for image_name in os.listdir(input_folder):
        if f'{image_name}' in file_names:
            src_path = os.path.join(input_folder, image_name)
            dst_path = os.path.join(output_folder, image_name)
            shutil.copy2(src_path, dst_path)

    list_of_tensors = []
    transform = transforms.Compose([transforms.PILToTensor()])

    for image_name in os.listdir(output_folder):

        image_path = os.path.join(output_folder, image_name)
        img = Image.open(image_path).convert("RGB")

        new_image = expand2square(img, (0, 0, 0)).resize((128,128))
        save_path = os.path.join(prepro_folder, image_name)

        new_image.save(save_path)

        transf = transform(new_image)
        tensor = transf.permute(1, 2, 0)
        list_of_tensors.append(tensor)

    X_prepro = np.array([tensor.numpy() for tensor in list_of_tensors])
    X_prepro = X_prepro / 255

    return X_prepro

def preprocess_y():

    dictio = {}

    excluded_id = set(images_plus_de_10['id'])


    for dict in data['annotations']:
        if dict['image_id'] not in excluded_id:
            dictio[dict['image_id']] = []

    for dict in data['annotations']:
        lst = []
        lst.append(dict['bbox'])
        lst.append(dict['category_id'])
        if dict['image_id'] not in images_plus_de_10['id']:
            dictio[dict['image_id']].append(lst)

    for key, value in dictio.items():
        for bb in value:
            bb[0][0] = (bb[0][0] /1920) * 128
            bb[0][2] = (bb[0][2] /1920) * 128
            bb[0][1] = (bb[0][1] /1080) * 128
            bb[0][3] = (bb[0][3] /1080) * 128

    for key, value in dictio.items():
        if len(value) < 10:
            while len(value) < 10:
                value.append([[0,0,0,0],0])

    y_bbox = np.zeros((2554, 10, 4))

    dataframe = pd.DataFrame(dictio)

    for column in range(2554):
            for i in range(10):
                bbox, class_id = dataframe.iloc[i, column]
                y_bbox[column, i] = bbox

    y_bbox = y_bbox/128

    y_class = np.zeros((2554, 10, 1))

    for column in range(2554):
            for i in range(10):
                bbox, class_id = dataframe.iloc[i, column]
                y_class[column, i] = class_id

    return y_bbox, y_class
