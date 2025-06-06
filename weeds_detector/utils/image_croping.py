import os
from PIL import Image
import json
import pandas as pd
from weeds_detector.data import get_filepath_in_directories, get_filepath, get_json_content
from weeds_detector.params import *
from google.cloud import storage
import requests
from io import BytesIO
from requests.exceptions import MissingSchema


def output_directory():
    """Output directory (cropped_images)"""
    output_dir = f"preprocessed/croped_{DATA_SIZE}"
    return output_dir

def load_data(json_data_path):
    """Load json path and csv path and load data from json"""
    csv_path = get_filepath("image_characteristics.csv")
    data = get_json_content(json_data_path)

    return csv_path,data


def load_id_to_filename(csv_path: str) -> dict:
    """Map image_id to filename from CSV file and
    take only DATA_SIZE numbers of images."""
    df = pd.read_csv(csv_path)
    size = DATA_SIZE_MAP.get(DATA_SIZE)
    if size is None:
        # prend tout
        df = df
    else:
        df = df.iloc[:size]
    id_to_filename = dict(zip(df['id'], df['filename']))
    return id_to_filename


def load_image(filename: str, image_dir: list) -> Image:
    """Load images from data.py with the image_path"""
    image_path = get_filepath_in_directories(filename, image_dir)
    if FILE_ORIGIN == 'local':
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {filename}")
        return Image.open(image_path), image_path
    elif FILE_ORIGIN == 'gcp':
        response = requests.get(image_path)
        if response.status_code != 200:
            raise FileNotFoundError(f"❌ Unable to download image from GCP URL: {image_path}")
        image = Image.open(BytesIO(response.content))
        return image, image_path


def crop_image(image: Image, bbox: list) -> Image:
    """Crop image with bbox from json file"""
    x, y, w, h = bbox
    return image.crop((int(x), int(y), int(x + w), int(y + h)))


def build_filename(filename, image_id, bbox_id, category_id):
    """The name of the output file"""
    return f"{filename}_{image_id}_{bbox_id}_{category_id}.png"

def save_cropped_image(cropped: Image, output_dir: str, output_name: str):
    """Save cropped images in a new file if FILE_ORIGIN is gcp or local"""
    if FILE_ORIGIN == 'local':
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        cropped.save(output_path)

    elif FILE_ORIGIN == 'gcp':
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_path = os.path.join("data", output_dir, output_name)
        blob = bucket.blob(blob_path)

        image_bytes = BytesIO()
        cropped.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        blob.upload_from_file(image_bytes, content_type='image/png')



def crop_annotations(data, id_to_filename, image_dir, output_dir):
    """
    Extracts image crops based on annotation data.

    Args:
        data (dict): JSON-like dictionary containing annotation information.
        id_to_filename (dict): Dictionary mapping {image_id: image_filename}.
        image_dir (str): Directory containing the input images.
        output_dir (str): Directory where cropped images will be saved.
    """
    count = 0
    valid_image_ids = set(id_to_filename.keys())

    # Filtrage préalable
    annotations = [
        ann for ann in data["annotations"]
        if ann["image_id"] in valid_image_ids
    ]
    for annotation in annotations:
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        bbox_id = annotation["id"]

        filename = id_to_filename[image_id]
        try:
            image, image_path = load_image(filename, image_dir)
            cropped = crop_image(image, bbox)
            output_name = build_filename(filename, image_id, bbox_id, category_id)
            save_cropped_image(cropped, output_dir, output_name)
            count += 1
            print(f"✅ {count} crops saved in '{output_dir}'")
        except (FileNotFoundError, MissingSchema) as e:
            print(f"❌ Skipping {filename} (ID: {image_id}) due to error: {e}")
