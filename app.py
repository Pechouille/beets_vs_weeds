import PIL.Image
import streamlit as st
import time
import os
import numpy as np
from PIL import Image, ImageDraw
from weeds_detector.utils.bbox_from_UNET import get_bbox_from_mask
import requests
import json
import base64
import io
import streamlit as st
from PIL import Image

TEMP_STATIC_IMAGE = "./static/temp.png"
API_URL = "https://beets-vs-weeds-api-prod-zpq6nq7z5q-od.a.run.app/predict"

def call_predict_API(model_name:str, uploadedFile:object) -> object:
    '''Simulate the call to the UNET segmentation computing API'''
    mask_image = None

    headers = {
        "accept": "application/json"
    }

    params = {
        "model":model_name
    }

    files = {
        "file": (uploadedFile.name, uploaded_file.getvalue(), "image/png")
    }
    response = requests.post(API_URL, params=params, files=files, headers=headers, verify=False)
    if "mask" in json.loads(response.content).keys():
        image_bytes  = base64.b64decode(json.loads(response.content)["mask"])
        mask_image = Image.open(io.BytesIO(image_bytes))
    bboxes = json.loads(response.content)["bboxes"]

    return mask_image, bboxes


st.title("Beets Vs Weeds: Segmentation and classification")

############################################################################
### Model selection ########################################################
MODEL_ALL_IN_ONE = "Convolution Neural Network only (CNN)"
MODEL_SEPARATED = "Segmentation UNET + Classification CNN"

front_end_label_to_model_selection = {
    MODEL_ALL_IN_ONE: "segm_classif",
    MODEL_SEPARATED: "unet"
}

model_option = st.selectbox(
    "Please select a model to use",
    (MODEL_ALL_IN_ONE, MODEL_SEPARATED),
)

st.write("You selected:", model_option)

############################################################################
### Image file selection ####################################################
uploaded_file = st.file_uploader("Select an image...", type=["png"])

# 📸 Affichage de l'image
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    #Display the original file
    st.image(image, caption=uploaded_file.name, use_container_width=True)

############################################################################
### Segmentation prediction on the image ###################################

    #result mask display
    st.text("Predicted mask:")
    with st.spinner("Computing segmentation and classification, please wait...", show_time=True):
        mask_image, bboxes = call_predict_API(front_end_label_to_model_selection[model_option], uploaded_file)
    if mask_image != None:
        mask_image = mask_image.resize(image.size)
        st.image(mask_image, caption=f"Mask predicted from {uploaded_file.name}", use_container_width = True)

############################################################################
### Compute the result against the original image ###########################

    #create the temp file
    with st.spinner("Applying mask to original picture...", show_time=True):
        bb_original = PIL.Image.open(uploaded_file)
        bb_original.save(TEMP_STATIC_IMAGE)
        bb_original = PIL.Image.open(TEMP_STATIC_IMAGE).convert("RGBA")
        for bbox in bboxes:
            inner_bbox = bbox['bbox']
            class_bb = int(bbox['class'])
            if class_bb == 0:
                color = (0,255,0,60)
            else :
                color = (255,0,0,60)
            bbbox_mask = Image.new('RGBA', bb_original.size, (0,0,0,0))
            draw = ImageDraw.Draw(bbbox_mask)
            draw.rectangle((inner_bbox[0], inner_bbox[1], inner_bbox[2], inner_bbox[3]), fill=color)
            bb_original = Image.alpha_composite(bb_original, bbbox_mask)

        bb_original.save(TEMP_STATIC_IMAGE)

    st.text("Bounding box on original picture:")
    st.image(TEMP_STATIC_IMAGE, caption=f"Bounding box from {uploaded_file.name}", use_container_width = True)
    ############################################################################
