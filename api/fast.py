import os
import uuid
from random import randint

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from weeds_detector.utils.display_bbox import display_image_with_bounding_boxes

# Dossiers pour stocker les images
UPLOAD_DIR = "api/uploads/"
OUTPUT_DIR = "api/outputs/"
os.makedirs(UPLOAD_DIR, exist_ok=True)
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
    extension = file.filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"filename": filename}

@app.get("/show/")
async def show_random_image_with_boxes():
    # Trouver une image jpg ou png dans le dossier uploads
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(('.jpg', '.png'))]
    if not files:
        return {"error": "No images found"}

    selected = files[randint(0, len(files) - 1)]
    image_path = os.path.join(UPLOAD_DIR, selected)
    output_path = os.path.join(OUTPUT_DIR, f"boxed_{selected}")

    # Crée une image avec bounding boxes
    display_image_with_bounding_boxes(image_path, save_path=output_path)

    return FileResponse(output_path, media_type="image/png")
