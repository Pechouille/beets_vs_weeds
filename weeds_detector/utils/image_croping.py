import os
from PIL import Image
import json
import pandas as pd

IMAGE_DIR = "data/all/"
OUTPUT_DIR = "croped_data_first_10/"
ANNOTATION_FILE = "data/json_train_set.json"
CSV_PATH = "data/image_characteristics.csv"


with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)

df = pd.read_csv(CSV_PATH)
id_to_filename = dict(zip(df['id'], df['filename']))


def crop_annotations(data, id_to_filename, image_dir, output_dir, max_images):
    """
    Extracts image crops based on annotation data.

    Args:
        data (dict): JSON-like dictionary containing annotation information.
        id_to_filename (dict): Dictionary mapping {image_id: image_filename}.
        image_dir (str): Directory containing the input images.
        output_dir (str): Directory where cropped images will be saved.
        max_images (int): Maximum number of annotations to process (for quick testing).
    """
    count = 0

    os.makedirs(output_dir, exist_ok=True)

    for ann in data["annotations"]:
        if count >= max_images:
            break

        image_id = ann["image_id"]
        bbox = ann["bbox"]  # [x_min, y_min, width, height]
        category_id = ann["category_id"]
        bbox_id = ann["id"]

        filename = id_to_filename.get(image_id)
        if filename is None:
            print(f"❌ image_id {image_id} not found in mapping.")
            continue

        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            print(f"❌ Image not found : {image_path}")
            continue

        try:
            image = Image.open(image_path)

            x, y, w, h = bbox
            left = int(x)
            top = int(y)
            right = int(x + w)
            bottom = int(y + h)

            cropped = image.crop((left, top, right, bottom))

            output_name = f"{filename}_{image_id}_{bbox_id}_{category_id}.png"
            output_path = os.path.join(output_dir, output_name)

            cropped.save(output_path)
            count += 1

        except FileNotFoundError:
            print(f"❌ File not found: {image_path}")
