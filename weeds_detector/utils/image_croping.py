import time
from PIL import Image
import pandas as pd
from weeds_detector.data import get_filepath_in_directories, get_filepath, get_json_content, get_existing_files
from weeds_detector.params import *
from requests.exceptions import MissingSchema
from weeds_detector.utils.images import save_image, load_image


def output_directory():
    """Output directory (cropped_images)"""
    output_dir = f"data/croped_images/croped_{DATA_SIZE}"
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


def crop_image(image: Image.Image, bbox: list) -> Image.Image:
    """Crop image with bbox from json file"""
    x, y, w, h = bbox
    return image.crop((int(x), int(y), int(x + w), int(y + h)))


def build_filename(filename: str, image_id: int, bbox_id: int, category_id: int) -> str:
    """The name of the output file"""
    return f"{filename}_{image_id}_{bbox_id}_{category_id}.png"

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
    print(f"Starting crop processing for {len(data['annotations'])} annotations")
    print(f"Output directory: {output_dir}")
    print(f"File origin: {FILE_ORIGIN}")

    # Get existing crops to avoid reprocessing
    existing_crops = get_existing_files(output_dir)
    print(f"Found {len(existing_crops)} existing crops to skip")

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
    print(f"Processing {total_annotations} valid annotations")

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
            print(f"Skipped {output_name} already processed crops")
            continue

        try:
            # Load and process image
            image, image_path = load_image(filename, image_dir)
            cropped_image = crop_image(image, bbox)
            save_image(cropped_image, output_dir, output_name)

            count += 1
            total_count += 1
            elapsed_time = time.time() - start_time
            rate = count / elapsed_time if elapsed_time > 0 else 0
            print(f"✅ {count} | {total_count} | Processed {filename} crops to {output_name} ({rate:.2f} crops/sec). ")

        except (FileNotFoundError, MissingSchema) as e:
            print(f"❌ 1 - Error processing {filename} (ID: {image_id}): {e}")

    # Final summary
    elapsed_time = time.time() - start_time
    rate = count / elapsed_time if elapsed_time > 0 else 0

    print(f"🎉 PROCESSING COMPLETE!")
    print(f"✅ Successfully processed: {count} crops")
    print(f"⏭️  Skipped (already processed): {skipped_count} crops")
    print(f"❌ Errors encountered: {error_count} crops")
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    print(f"📊 Processing rate: {rate:.2f} crops/second")
    print(f"📁 Output directory: {output_dir}")


def main(set_name: str):
    """Main execution function"""

    try:
        print("=" * 60)
        print("STARTING IMAGE CROPPING PROCESS")
        print("=" * 60)

        # Load data
        csv_path, data = load_data(set_name)
        id_to_filename = load_id_to_filename(csv_path)
        output_dir = output_directory()

        # Get image directories (assuming this comes from your params)
        image_dir = ["all"]  # You'll need to set this based on your setup

        print(f"Loaded {len(data['annotations'])} annotations")
        print(f"Loaded {len(id_to_filename)} image mappings")

        # Start processing
        crop_annotations(data, id_to_filename, image_dir, output_dir)

        print("=" * 60)
        print("IMAGE CROPPING PROCESS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"Fatal error in main process: {e}")
        raise


if __name__ == "__main__":
    main()
