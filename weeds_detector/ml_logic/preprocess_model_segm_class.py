import pandas as pd
import numpy as np
import json
import os
import shutil
import requests
from io import BytesIO
from torchvision import transforms
from PIL import Image
from weeds_detector.params import *
from google.cloud import storage

from weeds_detector.utils.padding import expand2square
from weeds_detector.data import get_filepath, get_json_content, get_all_files_path_and_name_in_directory, get_folderpath, get_existing_files

def df_img_selected_by_max_bbox_nbr(number_of_bbox, image_characteristics_filename):
    file_url = get_filepath(image_characteristics_filename)
    file_df = pd.read_csv(file_url)
    file_filtered_df = file_df[file_df.number_items_per_picture > number_of_bbox][['id', 'filename', 'number_items_per_picture']]

    return file_filtered_df

def annotated_img_ids(splited_data):
    annotated_ids = set(bbox["image_id"] for bbox in splited_data["annotations"])

    return annotated_ids

def excluded_filenames(file_filtered_df):
    excluded_filename = set(file_filtered_df["filename"])

    return excluded_filename

def img_needed_filenames(splited_data, excluded_filenames, annotated_img_ids):
    filenames = []
    for img in splited_data.get("images", []):
        file_name = img["file_name"]
        if file_name not in excluded_filenames and img["id"] in annotated_img_ids:
            filenames.append(file_name)
    img_needed = set(filenames)
    return img_needed


def create_folder(folder_name):
    if FILE_ORIGIN == 'local':
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        return folder_name
    elif FILE_ORIGIN == 'gcp':
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        folder_blob_name = folder_name if folder_name.endswith('/') else folder_name + '/'
        blob = bucket.blob(folder_blob_name)
        if not blob.exists():
            blob.upload_from_string('', content_type='application/x-www-form-urlencoded;charset=UTF-8')
        return blob.name

def copy_file(file_name, origin_dir, output_dir):
    if FILE_ORIGIN == 'local':
        file_path = get_filepath(file_name)
        dst_path = os.path.join(output_dir, file_name)
        shutil.copy2(file_path, dst_path)
        print("✅ Images copied in the ouput_dir")
        return None
    elif FILE_ORIGIN == 'gcp':
        storage_client = storage.Client()
        source_bucket = storage_client.bucket(BUCKET_NAME)
        source_blob = source_bucket.blob(os.path.join(origin_dir, file_name))
        destination_blob_name = os.path.join(output_dir, file_name)
        blob_copy = source_bucket.copy_blob(
                source_blob, source_bucket, destination_blob_name
        )
        print("✅ Images copied in the ouput_dir")
        return None


def preprocess_images(number_of_bbox, image_characteristics_filename = "image_characteristics.csv", data_split_filename = "json_train_set.json"):
    """
    input folder being the folder where all images are located
    output folder is the empty folder that will contain only the images we need to preprocess
    prepro folder is the empty folder that will contain the images preprocessed
    """
    print("1 - START PREPROCESS IMAGE")
    print("---------------------------")

    print("2 - START LOAD DATA")
    print("---------------------------")
    splited_data = get_json_content(data_split_filename)

    file_filtered_df = df_img_selected_by_max_bbox_nbr(number_of_bbox, image_characteristics_filename)

    excluded_filename = excluded_filenames(file_filtered_df)
    annotated_ids = annotated_img_ids(splited_data)

    img_needed = img_needed_filenames(splited_data, excluded_filename, annotated_ids)
    print("3 - DATA LOADED")
    print("---------------------------")
    list_of_tensors = []
    transform = transforms.Compose([transforms.PILToTensor()])
    print("4 - START PREPROCESS EACH IMAGES")
    print("---------------------------")
    count = 0
    for file_path, file_name in get_all_files_path_and_name_in_directory("all", extensions = [".png"]):
        print(f"Start Preprocess : {file_name}")
        print("---------------------------")
        if file_name in img_needed:
            response = requests.get(file_path)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            resized_value = int(RESIZED)
            new_image = expand2square(img, (0, 0, 0)).resize((resized_value, resized_value))
            output_dir2 = create_folder('images_preprocessed')
            save_path = get_folderpath(output_dir2)
            new_image.save(save_path)
            transf = transform(new_image)
            tensor = transf.permute(1, 2, 0)
            list_of_tensors.append(tensor)
            count +=1
        print(f"{count} / {len(img_needed)} Image Preprocessed : {file_name}")
        print("---------------------------")
    X_prepro = np.array([tensor.numpy() for tensor in list_of_tensors])
    X_prepro = X_prepro / 255

    return X_prepro

def preprocess_y(number_of_bbox, image_characteristics_filename = "image_characteristics.csv", data_split_filename = "json_train_set.json"):

    splited_data = get_json_content(data_split_filename)

    file_filtered_df = df_img_selected_by_max_bbox_nbr(number_of_bbox, image_characteristics_filename)

    dictio = {}

    excluded_id = set(file_filtered_df['id'])

    for dict in splited_data['annotations']:
        if dict['image_id'] not in excluded_id:
            dictio[dict['image_id']] = []

    for dict in splited_data['annotations']:
        lst = []
        lst.append(dict['bbox'])
        lst.append(dict['category_id'])
        if dict['image_id'] not in file_filtered_df['id']:
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
