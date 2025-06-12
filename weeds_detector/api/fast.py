# Imports de bibliothèques standard
import os
import io
import base64
from contextlib import asynccontextmanager

# Imports de bibliothèques tierces
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Imports locaux
from weeds_detector.utils.display_bbox import api_display_image_with_bounding_boxes, load_bounding_boxes
from weeds_detector.utils.image_croping import crop_image
from weeds_detector.ml_logic.registry import load_model
from weeds_detector.ml_logic.preprocess_model_class import preprocess_features, preprocess_single_image
from weeds_detector.params import *
from weeds_detector.utils.padding import expand2square
from weeds_detector.utils.pipeline_mask_unet_api import (
    prepare_image_for_unet,
    predict_mask,
    crop_from_mask_and_save
)
from weeds_detector.ml_logic.model_UNET import dice_loss, dice_coeff, combined_loss

custom_objects = {
    "compile_metrics": dice_coeff,
    "loss": combined_loss,
}

# Dossiers pour stocker les images
UPLOAD_DIR = "data/all/"
OUTPUT_DIR = "weeds_detector/api/outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()
app.state.model_class = load_model(model_type = 'cnn_classif')
app.state.model_segm_classif = load_model(model_type = 'cnn_segm_classif_final2')
app.state.model_unet_test = load_model(model_type = 'unet_segmentation_model', custom_objects=custom_objects)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """
    Point de terminaison de base pour vérifier si l'API est en cours d'exécution.

    Returns:
        dict: Un dictionnaire avec un message de bienvenue.
    """
    return {"message": "Hello, API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Query(..., enum=["unet", "segm_classif"])):
    """
    Prédit les résultats pour une image téléchargée en utilisant le modèle spécifié.

    Args:
        file (UploadFile): L'image téléchargée.
        model (str): Le modèle à utiliser pour la prédiction ("unet" ou "segm_classif").

    Returns:
        dict: Un dictionnaire contenant les résultats de la prédiction.
    """
    try:
        # Lire et ouvrir l’image
        uploaded_image = await file.read()
        image_pil = Image.open(io.BytesIO(uploaded_image)).convert("RGB")
        original_size = image_pil.size

        response = {}

        if model == "unet":
            # Preprocess UNET
            image_tensor = prepare_image_for_unet(image_pil)
            mask_bin = predict_mask(app.state.model_unet_test, image_tensor)

            # Convertir 0/1 en 0/255
            mask_uint8 = (mask_bin * 255).astype(np.uint8)

            # Créer une image PIL à partir du masque
            mask_img = Image.fromarray(mask_uint8)

            # Sauver dans un buffer
            buffer = io.BytesIO()
            mask_img.save(buffer, format="PNG")
            buffer.seek(0)
            base64_mask = base64.b64encode(buffer.getvalue()).decode('utf-8')
            # Crops
            save_dir = "data/croped_images_UNET"
            unet_bboxs, images_crops = crop_from_mask_and_save(image_pil, mask_bin, original_size, save_dir, file.filename)
            for unet_bbox, image_crop in zip(unet_bboxs, images_crops):
                X_processed = preprocess_single_image(image_crop["image_crop"])
                crop_classif = app.state.model_class.predict(X_processed)
                unet_bbox["class"] = round(crop_classif[0][0])
            response = {
                "mask": base64_mask,
                "bboxes": unet_bboxs
            }

        if model == "segm_classif":
            X_processed_segm_class = preprocess_single_image(image_pil)
            segm_classif_class, segm_classif_bbox = app.state.model_segm_classif.predict(X_processed_segm_class)
            segm_classif_bbox = segm_classif_bbox * 256
            for bbox in segm_classif_bbox:
                bbox[0] = (bbox[0]/256) * 1980
                bbox[2] = (bbox[2]/256) * 1980
                bbox[1] = (bbox[1]/256) * 1080
                bbox[3] = (bbox[3]/256) * 1080
            segm_classif_bboxs = []
            for class_bbox, segm_bbox in zip(segm_classif_class[0], segm_classif_bbox[0]):
                x, y, w, h = segm_bbox
                segm_classif_bboxs.append({
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "class": int(round(class_bbox[0]))
                })

            response = {
                "bboxes": segm_classif_bboxs
            }

        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/base")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Télécharge une image et génère une image annotée avec des boîtes englobantes.

    Args:
        file (UploadFile): L'image téléchargée.

    Returns:
        FileResponse: L'image annotée avec des boîtes englobantes.
    """
    filename = file.filename  # Ne pas renommer
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
       f.write(await file.read())

    # Générer l’image annotée
    output_filename = f"boxed_{filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
       api_display_image_with_bounding_boxes(file_path, save_path=output_path)
    except FileNotFoundError:
       return {"error": f"Fichier XML non trouvé pour {filename}. Attendait : data/all/{os.path.splitext(filename)[0]}.xml"}
    except Exception as e:
       return {"error": str(e)}
    return FileResponse(output_path, media_type="image/png")
