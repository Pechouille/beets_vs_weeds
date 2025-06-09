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


# @app.get("/show/")
# async def show_random_image_with_boxes():
#     # Trouver une image jpg ou png dans le dossier uploads
#     files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(('.jpg', '.png'))]
#     if not files:
#         return {"error": "No images found"}

#     selected = files[randint(0, len(files) - 1)]
#     image_path = os.path.join(UPLOAD_DIR, selected)
#     output_path = os.path.join(OUTPUT_DIR, f"boxed_{selected}")

#     # Crée une image avec bounding boxes
#     api_display_image_with_bounding_boxes(image_path, save_path=output_path)

#     return FileResponse(output_path, media_type="image/png")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # 🔁 Entraînement du modèle au démarrage de l'API
#     print("📦 Initialisation du modèle...")

#     # Ici tu peux charger tes vraies données X, y si tu veux
#     X_dummy = np.random.rand(100, 128, 128, 3)
#     y_dummy = np.random.randint(0, 2, 100)

#     model = initialize_model()
#     model = compile_model(model)
#     model, _ = train_model(model, X_dummy, y_dummy)

#     app.state.model = model

#     print("✅ Modèle prêt à prédire !")
#     yield  # <-- l'API démarre ici

#     # 🧹 Code de shutdown éventuel (libération de ressources, etc.)
#     print("🛑 Arrêt de l'API...")

# # 👇 Crée l'application FastAPI avec le gestionnaire de vie
# app = FastAPI(lifespan=lifespan)

# @app.post("/predict/")
# async def predict(image: UploadFile = File(...)):
#     # Lire l'image
#     contents = await image.read()
#     with open("temp_image.jpg", "wb") as f:
#         f.write(contents)

#     # Prétraitement : redimensionne et normalise
#     img = load_img("temp_image.jpg", target_size=(128, 128))
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)  # forme: (1, 128, 128, 3)

#     # Prédiction
#     model = app.state.model
#     prediction = model.predict(img_array)[0][0]
#     class_pred = int(prediction > 0.5)

#     label = "betterave" if class_pred == 1 else "mauvaise herbe"
#     return {"prediction": float(prediction), "classe": label}
