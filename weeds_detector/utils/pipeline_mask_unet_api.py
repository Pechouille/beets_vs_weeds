import io
from PIL import Image
import numpy as np
import os
import requests
from tensorflow import io as tf_io
from tensorflow import image as tf_image
from tensorflow import cast, float32, expand_dims

from weeds_detector.utils.images import save_image
from weeds_detector.utils.bbox_from_UNET import get_bbox_from_mask
from weeds_detector.data import get_filepath


def prepare_image_for_unet(pil_image: Image.Image, target_size=(256, 256)) -> np.ndarray:
    """
    Prépare une image PIL pour la prédiction UNet.
    """
    pil_image = pil_image.resize(target_size)
    image_array = np.array(pil_image).astype("float32") / 255.0
    image_tensor = expand_dims(image_array, axis=0)
    return image_tensor


def predict_mask(model, prepared_tensor: np.ndarray, threshold=0.9) -> np.ndarray:
    """
    Prédit un masque binaire à partir d un modèle UNet.
    """
    y_pred = model.predict(prepared_tensor, verbose=0)
    y_pred_binary = (y_pred[0, :, :, 0] > threshold).astype(np.uint8)
    return y_pred_binary


def crop_from_mask_and_save(pil_image: Image.Image, mask_binary: np.ndarray,
                            original_size: tuple, save_dir: str, filename: str) -> list:
    """
    Calcule les bounding boxes, crop les objets, les sauvegarde,
    et retourne la liste des BBoxes.
    """
    os.makedirs(save_dir, exist_ok=True)

    bboxes = get_bbox_from_mask(
        y_pred_binary=mask_binary,
        resized_size=(256, 256),
        original_size=original_size
    )

    results = []
    for i, (x, y, w, h) in enumerate(bboxes):
        xmin, ymin, xmax, ymax = int(x), int(y), int(x + w), int(y + h)
        crop = pil_image.crop((xmin, ymin, xmax, ymax))
        crop_filename = f"{os.path.splitext(filename)[0]}_crop_{i}.png"
        save_image(crop, save_dir, crop_filename)

        results.append({
            "bbox_id": i,
            "bbox": [xmin, ymin, xmax, ymax],
            "class": ""  # à remplir plus tard si besoin
        })

    return results
