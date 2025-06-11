import PIL.Image
import streamlit as st
import time
import os
import numpy as np
from PIL import Image, ImageDraw
from weeds_detector.utils.bbox_from_UNET import get_bbox_from_mask



import streamlit as st
from PIL import Image

pred_selector = {
    "near30_near30_01_06_2021_v_0_8.png": "y_pred0.png",
    "near30_near30_01_06_2021_v_1_2.png" : "y_pred1.png",
    "near30_near30_01_06_2021_v_1_43.png" : "y_pred2.png",
    "near30_near30_01_06_2021_v_2_65.png" : "y_pred3.png",
    "near30_near30_01_06_2021_v_2_66.png" : "y_pred4.png"
}

TEMP_STATIC_IMAGE = "./static/temp.png"

def call_predict_API(model_name:str, image_path:str) -> object:
    '''Simulate the call to the UNET segmentation computing API'''
    '''TODO: replace content by a call to the corresponding API'''
    time.sleep(10) # place for the UNET predict
    mask_image_name = pred_selector[image_path]
    mask_image = Image.open(os.path.join("./static/", mask_image_name))

    return mask_image


st.title("Beets Vs Weeds: Segmentation and classification")

############################################################################
### Model selection ########################################################
MODEL_ALL_IN_ONE = "Segmentation and classification all in one"
MODEL_SEPARATED = "Segmentation then classification"

model_option = st.selectbox(
    "Please select a model to use",
    (MODEL_ALL_IN_ONE, MODEL_SEPARATED),
)

st.write("You selected:", model_option)

############################################################################
### Image file selection ####################################################
uploaded_file = st.file_uploader("Choisissez une image...", type=["png"])

# 📸 Affichage de l'image
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    #Display the original file
    st.image(image, caption=uploaded_file.name, use_container_width=True)

############################################################################
### Segmentation prediction on the image ####################################

    #result mask display
    st.text("Predicted mask:")
    with st.spinner("Computing segmentation, please wait...", show_time=True):
        mask_image = call_predict_API(model_option, uploaded_file.name)
    st.image(mask_image, caption=f"Mask predicted from {uploaded_file.name}", use_container_width = True)

############################################################################
### Compute the result against the original image ###########################

    #display the original file with the bounding bow generated from the mask
    with st.spinner("Computing bounding boxes from mask...", show_time=True):
        image_binary = np.asarray(PIL.Image.open(os.path.join("./static/",  pred_selector[uploaded_file.name])))
        bounding_boxes = get_bbox_from_mask(image_binary, resized_size=(1920,1080), original_size=(1920,1080))

    #create the temp file
    with st.spinner("Applying mask to original picture...", show_time=True):
        bb_original = PIL.Image.open(uploaded_file)
        bb_original.save(TEMP_STATIC_IMAGE)
        bb_original = PIL.Image.open(TEMP_STATIC_IMAGE).convert("RGBA")
        for bbox in bounding_boxes:
            if len(bbox) < 4:
                st.title(f"ERROR: expected len of bbox=4 but was {len(bbox)}, {bbox}")
                break

            bbbox_mask = Image.new('RGBA', bb_original.size, (0,0,0,0))
            draw = ImageDraw.Draw(bbbox_mask)
            draw.rectangle((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]), fill=(0, 0, 255,60))
            bb_original = Image.alpha_composite(bb_original, bbbox_mask)

        bb_original.save(TEMP_STATIC_IMAGE)

    st.text("Bounding box on original picture:")
    st.image(TEMP_STATIC_IMAGE, caption=f"Bounding box from {uploaded_file.name}", use_container_width = True)
    ############################################################################
