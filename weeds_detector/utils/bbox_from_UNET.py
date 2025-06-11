from tensorflow import io
from tensorflow import image
from tensorflow import cast
from tensorflow import expand_dims
from tensorflow import float32
import numpy as np
from skimage.measure import label, regionprops, find_contours
import os
from PIL import Image
from weeds_detector.data import get_all_files_path_and_name_in_directory, get_filepath
from weeds_detector.utils.images import save_image
import requests
from io import BytesIO


def process_test_image(image_path):
    png = io.read_file(image_path)
    png = image.decode_image(png, channels=3)
    png = image.resize(png, [256, 256])
    png = cast(png, float32) / 255.0
    png = expand_dims(png, axis=0)  # batch size de 1
    return png

def prediction_mask_image(model, image):
    y_pred = model.predict(image, verbose=0)
    y_pred_binary = (y_pred[0, :, :, 0] > 0.9).astype(np.uint8)
    return y_pred_binary

def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))
    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            if 0 <= x < h and 0 <= y < w:
                border[x, y] = 255
    return border

def mask_to_bbox(mask):
    bboxes = []
    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x = prop.bbox[1]
        y = prop.bbox[0]
        width = prop.bbox[3] - prop.bbox[1]
        height = prop.bbox[2] - prop.bbox[0]
        bboxes.append([x, y, width, height])
    return bboxes

def get_bbox_from_mask(y_pred_binary, resized_size=(256, 256), original_size=(1920, 1080)):
    """
    Transforms a binary mask into bounding boxes and rescales them to the original image dimensions.

    Args:
        y_pred_binary (np.ndarray): binary prediction mask.
        resized_size (tuple): size of the mask (width, height) used as input to the model.
        original_size (tuple): original size of the image (width, height).

    Returns:
        list: list of bounding boxes in the format [x, y, width, height], rescaled to the original size.
    """

    y_pred_binary_255 = y_pred_binary * 255
    bboxes = mask_to_bbox(y_pred_binary_255)

    scale_x = original_size[0] / resized_size[0]
    scale_y = original_size[1] / resized_size[1]

    rescaled_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        x *= scale_x
        y *= scale_y
        w *= scale_x
        h *= scale_y
        rescaled_bboxes.append([x, y, w, h])

    return rescaled_bboxes

# ----------- Pipeline complet -----------

def predict_all_images(model, image_folder):
    """
    Predict bounding boxes for all .png images in a folder (GCP or local)
    """
    image_paths_info = get_all_files_path_and_name_in_directory(image_folder, extensions=[".png"])
    if image_paths_info is None:
        return {}

    results = {}
    for image_path, filename in image_paths_info:
        image = process_test_image(image_path)
        mask_bin = prediction_mask_image(model, image)
        bboxes = get_bbox_from_mask(mask_bin)
        results[filename] = bboxes

    return results

def crop_images_from_result(results: dict, image_dir: str):
    """
    Crop and save sub-images from original full-resolution images,
    either locally or to GCP, depending on FILE_TARGET.
    """
    crop_id = 0
    folder_name = f"data/croped_images_UNET/"

    for filename, bboxes in results.items():
        full_image_path = get_filepath(os.path.join(image_dir, filename))

        if full_image_path is None:
            print(f"❌ Image not found: {filename}")
            continue

        try:
            if full_image_path.startswith("http"):
                response = requests.get(full_image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(full_image_path)
        except Exception as e:
            print(f"⚠️ Error opening image {filename}: {e}")
            continue

        for i, bbox in enumerate(bboxes):
            x, y, w, h = map(int, bbox)
            crop = image.crop((x, y, x + w, y + h))

            crop_filename = f"{os.path.splitext(filename)[0]}_crop_{i}.png"
            save_image(crop, folder_name, crop_filename)

            crop_id += 1

    print(f"✅ {crop_id} crops saved at {folder_name}")
