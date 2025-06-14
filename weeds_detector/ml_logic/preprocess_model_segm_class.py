import pandas as pd
import numpy as np
import os
import shutil
import requests
from io import BytesIO
from tensorflow.keras.utils import img_to_array, save_img
from PIL import Image
from weeds_detector.params import FILE_ORIGIN, BUCKET_NAME, RESIZED
from google.cloud import storage

from weeds_detector.utils.padding import expand2square
from weeds_detector.data import get_filepath, get_json_content, get_all_files_path_and_name_in_directory, get_folderpath, get_existing_files
from weeds_detector.utils.images import save_image

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
        blob = bucket.blob(os.path.join("data", folder_blob_name))
        folder_exist = blob.exists()
        if not folder_exist:
            blob.upload_from_string('', content_type='application/x-www-form-urlencoded;charset=UTF-8')
        return blob.name, folder_exist

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
        source_bucket.copy_blob(
                source_blob, source_bucket, destination_blob_name
        )
        print("✅ Images copied in the ouput_dir")
        return None

def transform_image(file_name, file_path, output_dir):
    print(f"Create im age : preprocessed_{file_name} save in bucket {output_dir}")
    response = requests.get(file_path)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    resized_value = int(RESIZED)
    new_image = expand2square(img, (0, 0, 0)).resize((resized_value, resized_value))
    save_image(new_image, output_dir, f"preprocessed_{file_name}")
    return new_image

def preprocess_images(number_of_bbox, image_characteristics_filename = "image_characteristics.csv", data_split_filename = "json_train_set.json"):

    print("1 - START PREPROCESS IMAGE")
    print("---------------------------")

    print("2 - START ESTABLSIHING EXCLUDED IMAGES")
    print("---------------------------")

    splited_data = get_json_content(data_split_filename)

    file_filtered_df = df_img_selected_by_max_bbox_nbr(number_of_bbox, image_characteristics_filename)

    excluded_filename = excluded_filenames(file_filtered_df)
    annotated_ids = annotated_img_ids(splited_data)

    img_needed = img_needed_filenames(splited_data, excluded_filename, annotated_ids)

    print("3 - EXCLUDED IMAGES ESTABLISHED")
    print("---------------------------")

    list_of_tensors = []
    filenames_ordered = []

    print("4 - START PREPROCESS EACH IMAGES")
    print("---------------------------")

    count = 0
    output_dir, folder_exist = create_folder(f'images_preprocessed/full_images_resized/{RESIZED}x{RESIZED}')
    storage_client = storage.Client()
    source_bucket = storage_client.bucket(BUCKET_NAME)
    for file_path, file_name in get_all_files_path_and_name_in_directory("all", extensions = [".png"]):

        print(f"Start Preprocess : {file_name}")
        print("---------------------------")

        if file_name in img_needed:

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
            filenames_ordered.append(file_name)
            count +=1

        print(f"{count} / {len(img_needed)} Image Preprocessed : {file_name}")
        print("---------------------------")

    X_prepro = np.stack(list_of_tensors, axis=0)


    print("5 - PREPROCESS OF EACH IMAGE DONE")

    return X_prepro, filenames_ordered

def preprocess_y(filenames_ordered, number_of_bbox, image_characteristics_filename = "image_characteristics.csv", data_split_filename = "json_train_set.json"):

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

    resized = int(RESIZED)
    for key, value in dictio.items():
        for bb in value:
            bb[0][0] = (bb[0][0] / 1920) * resized
            bb[0][2] = (bb[0][2] / 1920) * resized
            bb[0][1] = (bb[0][1] / 1080) * resized
            bb[0][3] = (bb[0][3] / 1080) * resized

    for key, value in dictio.items():
        if len(value) < number_of_bbox:
            while len(value) < number_of_bbox:
                value.append([[0, 0, 0, 0], 0])

    number_of_images = len(filenames_ordered)
    y_bbox = np.zeros((number_of_images, number_of_bbox, 4))
    y_class = np.zeros((number_of_images, number_of_bbox, 1))

    filename_to_id = {img["file_name"]: img["id"] for img in splited_data["images"]}

    for idx, fname in enumerate(filenames_ordered):
        image_id = filename_to_id[fname]
        for i in range(number_of_bbox):
            bbox, class_id = dictio[image_id][i]
            y_bbox[idx, i] = bbox
            y_class[idx, i] = class_id

    y_bbox = y_bbox / resized

    return y_bbox, y_class
