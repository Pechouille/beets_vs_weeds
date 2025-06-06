import os
import time
import logger
from PIL import Image
import pandas as pd
from weeds_detector.data import get_filepath_in_directories, get_filepath, get_json_content, get_existing_files
from weeds_detector.params import *
from google.cloud import storage
import requests
from io import BytesIO
from requests.exceptions import MissingSchema
from typing import Set, Dict, List, Tuple
from weeds_detector.utils.logger import setup_logging


def output_directory():
    """Output directory (cropped_images)"""
    output_dir = f"preprocessed/croped_{DATA_SIZE}"
    return output_dir


def load_data(set_name: str):
    """Load json path and csv path and load data from json"""
    csv_path = get_filepath("image_characteristics.csv")
    data = get_json_content(f"json_{set_name}_set.json")
    return csv_path, data


def load_id_to_filename(csv_path: str) -> dict:
    """Map image_id to filename from CSV file and
    take only DATA_SIZE numbers of images."""
    df = pd.read_csv(csv_path)
    size = DATA_SIZE_MAP.get(DATA_SIZE)
    if size is None:
        df = df
    else:
        df = df.iloc[:size]
    id_to_filename = dict(zip(df['id'], df['filename']))
    return id_to_filename


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


def crop_image(image: Image.Image, bbox: list) -> Image.Image:
    """Crop image with bbox from json file"""
    x, y, w, h = bbox
    return image.crop((int(x), int(y), int(x + w), int(y + h)))


def build_filename(filename: str, image_id: int, bbox_id: int, category_id: int) -> str:
    """The name of the output file"""
    return f"{filename}_{image_id}_{bbox_id}_{category_id}.png"


def save_cropped_image(cropped: Image.Image, output_dir: str, output_name: str):
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

def crop_annotations(data: dict, id_to_filename: dict, image_dir: list, output_dir: str):
    """
    Extracts image crops based on annotation data.
    Enhanced with logging and skip logic for already processed files.

    Args:
        data (dict): JSON-like dictionary containing annotation information.
        id_to_filename (dict): Dictionary mapping {image_id: image_filename}.
        image_dir (list): Directory containing the input images.
        output_dir (str): Directory where cropped images will be saved.
    """
    logger.info(f"Starting crop processing for {len(data['annotations'])} annotations")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"File origin: {FILE_ORIGIN}")

    # Get existing crops to avoid reprocessing
    existing_crops = get_existing_files(output_dir)
    logger.info(f"Found {len(existing_crops)} existing crops to skip")

    count = 0
    total_count = len(existing_crops)
    skipped_count = 0
    error_count = 0
    valid_image_ids = set(id_to_filename.keys())

    # Filter annotations for valid image IDs
    annotations = [
        ann for ann in data["annotations"]
        if ann["image_id"] in valid_image_ids
    ]

    total_annotations = len(annotations)
    logger.info(f"Processing {total_annotations} valid annotations")

    start_time = time.time()
    existing_crops = get_existing_files(output_dir)
    for i, annotation in enumerate(annotations):
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        bbox_id = annotation["id"]

        filename = id_to_filename[image_id]
        output_name = build_filename(filename, image_id, bbox_id, category_id)

        # Skip if already processed
        if output_name in existing_crops:
            skipped_count += 1
            logger.info(f"Skipped {output_name} already processed crops")
            continue

        try:
            # Load and process image
            image, image_path = load_image(filename, image_dir)
            cropped = crop_image(image, bbox)
            save_cropped_image(cropped, output_dir, output_name)

            count += 1
            total_count += 1
            elapsed_time = time.time() - start_time
            rate = count / elapsed_time if elapsed_time > 0 else 0
            logger.info(f"✅ {count} | {total_count} | Processed {filename} crops to {output_name} ({rate:.2f} crops/sec). ")

        except (FileNotFoundError, MissingSchema) as e:
            logger.error(f"❌ 1 - Error processing {filename} (ID: {image_id}): {e}")

    # Final summary
    elapsed_time = time.time() - start_time
    rate = count / elapsed_time if elapsed_time > 0 else 0

    logger.info(f"🎉 PROCESSING COMPLETE!")
    logger.info(f"✅ Successfully processed: {count} crops")
    logger.info(f"⏭️  Skipped (already processed): {skipped_count} crops")
    logger.info(f"❌ Errors encountered: {error_count} crops")
    logger.info(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    logger.info(f"📊 Processing rate: {rate:.2f} crops/second")
    logger.info(f"📁 Output directory: {output_dir}")


def main(set_name: str):
    """Main execution function"""
    global logger
    logger = setup_logging()

    try:
        logger.info("=" * 60)
        logger.info("STARTING IMAGE CROPPING PROCESS")
        logger.info("=" * 60)

        # Load data
        csv_path, data = load_data(set_name)
        id_to_filename = load_id_to_filename(csv_path)
        output_dir = output_directory()

        # Get image directories (assuming this comes from your params)
        image_dir = ["all"]  # You'll need to set this based on your setup

        logger.info(f"Loaded {len(data['annotations'])} annotations")
        logger.info(f"Loaded {len(id_to_filename)} image mappings")

        # Start processing
        crop_annotations(data, id_to_filename, image_dir, output_dir)

        logger.info("=" * 60)
        logger.info("IMAGE CROPPING PROCESS COMPLETED")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        raise


if __name__ == "__main__":
    main()
