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


def crop_annotations(data, id_to_filename, image_dir, output_dir, max_images=5):
    """
    Extrait les crops des images à partir des annotations.

    Args:
        data (dict): Fichier json avec les annotations.
        id_to_filename (dict): Dictionnaire {image_id: nom_fichier_image}.
        image_dir (str): Répertoire contenant les images d'entrée.
        output_dir (str): Répertoire où sauvegarder les images cropées.
        max_images (int): Nombre maximum d'annotations à traiter (pour test rapide).
    """
    count = 0

    os.makedirs(output_dir, exist_ok=True)

    for ann in data["annotations"]:
        if count >= max_images:
            break

        image_id = ann["image_id"]
        bbox = ann["bbox"]  # [x_min, y_min, largeur, hauteur]
        category_id = ann["category_id"]
        bbox_id = ann["id"]

        filename = id_to_filename.get(image_id)
        if filename is None:
            print(f"❌ image_id {image_id} introuvable dans le mapping.")
            continue

        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            print(f"❌ Image non trouvée : {image_path}")
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

        except Exception as e:
            print(f"⚠️ Erreur avec {filename} (ID {image_id}): {e}")
