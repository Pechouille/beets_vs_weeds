import tensorflow as tf
import numpy as np
from skimage.measure import label, regionprops, find_contours


def process_test_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # couleur
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

def prediction_mask_image(model, image):
    y_pred = model.predict(image)
    y_pred_binary = (y_pred[0, :, :, 0] > 0.9).astype(np.uint8)

    return y_pred, y_pred_binary


def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x = prop.bbox[1]  # x1
        y = prop.bbox[0]  # y1
        width = prop.bbox[3] - prop.bbox[1]  # x2 - x1
        height = prop.bbox[2] - prop.bbox[0]  # y2 - y1

        bboxes.append([x, y, width, height])

    return bboxes

def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


def get_bbox_from_mask(y_pred_binary):
    y_pred_binary_255 = y_pred_binary * 255
    bboxes = mask_to_bbox(y_pred_binary_255)
    return bboxes
