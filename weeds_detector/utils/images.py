import os
from PIL import Image
from weeds_detector.data import get_filepath_in_directories
from weeds_detector.params import *
from google.cloud import storage
import requests
from io import BytesIO
from typing import Tuple

def load_image(filename: str, image_dir: list) -> Tuple[Image.Image, str]:
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

def save_image(image: Image.Image, output_dir: str, output_name: str):
    """Save cropped images in a new file if FILE_ORIGIN is gcp or local"""
    if FILE_ORIGIN == 'local':
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        image.save(output_path)

    elif FILE_ORIGIN == 'gcp':
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_path = os.path.join(output_dir, output_name)
        blob = bucket.blob(blob_path)

        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        blob.upload_from_file(image_bytes, content_type='image/png')
