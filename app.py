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

st.markdown(
    """
    <h1 style='
        text-align: center;
        color: white;
        font-size: 4em;
        margin-top: 4rem;
    '>
        Beets Vs Weeds: Segmentation and classification
    </h1>
    """,
    unsafe_allow_html=True
)

#https://plantura.garden/uk/wp-content/uploads/sites/2/2022/08/beetroot-in-ground.jpg
#https://www.suedzucker.com/wp-content/uploads/2021/11/sugar_beet_field.jpg


st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.suedzucker.com/wp-content/uploads/2021/11/sugar_beet_field.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Optional: add a subtle dark overlay for better contrast */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background: rgba(0, 0, 0, 0.4);  /* adjust opacity */
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Titre principal */
    h1 {
        color: white !important;
    }
    p {
    color: white !important;
    }

    /* Labels du selectbox et file uploader */
    label {
        color: white !important;
    }

    /* Paragraphes de st.write */
    .stMarkdown p {
        color: white !important;
    }

    /* Fond global si besoin */
    .main {
        background-color: #0e1117;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Réduire ou augmenter la largeur du conteneur principal */
    .block-container {
        max-width: 1200px;  /* Tu peux mettre 1000px, 80%, etc. */
        margin: 0 auto;     /* Centrer horizontalement */
        padding: 2rem;      /* Espace autour */
    }
    </style>
    """,
    unsafe_allow_html=True
)
############################################################################
### Model selection ########################################################
MODEL_ALL_IN_ONE = "Convolution Neural Network only (CNN)"
MODEL_SEPARATED = "Segmentation UNET + Classification CNN"

front_end_label_to_model_selection = {
    MODEL_ALL_IN_ONE: "segm_classif",
    MODEL_SEPARATED: "unet"
}

st.markdown(
    "<p style='color: white; font-size: 2em;'>Please select a model to use</p>",
    unsafe_allow_html=True
)
model_option = st.selectbox(
    "",  # Pas de label ici, car on l’a déjà mis au-dessus
    (MODEL_ALL_IN_ONE, MODEL_SEPARATED),
)

st.markdown(
    f"<p style='color: white; font-size: 1.2em;'>You selected: <strong>{model_option}</strong></p>",
    unsafe_allow_html=True
)

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
    st.markdown('<p style="color: white; font-size: 1.2em;">Predicted mask:</p>', unsafe_allow_html=True)
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

    st.markdown('<p style="color:white; font-size: 1.2em;">Bounding box on original picture:</p>', unsafe_allow_html=True)
    st.image(TEMP_STATIC_IMAGE, caption=f"Bounding box from {uploaded_file.name}", use_container_width = True)
    ############################################################################
