import os
import io
import base64
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse


from weeds_detector.utils.display_bbox import api_display_image_with_bounding_boxes, load_bounding_boxes
from weeds_detector.utils.image_croping import crop_image
from weeds_detector.ml_logic.registry import load_model
from weeds_detector.ml_logic.preprocess_model_class import preprocess_features, preprocess_single_image

# Dossiers pour stocker les images
UPLOAD_DIR = "data/all/"
OUTPUT_DIR = "api/outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hello, API is running."}

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
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




# @app.get("/predict/")
# async def predict(file: UploadFile = File(...)):
#     # Lire l'image uploadée
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert("RGB")

#     # Prétraitement
#     X = preprocess_single_image(image)

#     # Prédiction
#     app.state.model = load_model()
#     pred = app.state.model.predict(X)[0][0]  # sortie sigmoide entre 0 et 1

#     # Seuil à 0.5 par défaut pour classer
#     classe = int(pred > 0.5)

#     return {"prediction": classe, "confidence": float(pred)}
