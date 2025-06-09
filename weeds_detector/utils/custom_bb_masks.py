
import pandas as pd
from PIL import Image
import json
import numpy as np
import os



def get_image_bbox_annotation(data, bbox_id):
    returned_annotation = None
    for annotation in data:
        if annotation["id"] == bbox_id:
            #we found the bounding box !
            returned_annotation = annotation
    return returned_annotation


def create_empty_mask(width, height, output_path):
    mask = Image.new('RGB', (width, height), (0,0,0))
    #saving new mask
    mask.save(output_path)
    return mask

def generate_custom_bbox_mask(bounding_box_characterist_CSV_file,
                                        all_fields_lincolnbeet_JSON_file,
                                        input_directory,
                                        output_directory,
                                        max_image_generated = None):

    #If output directory does noçt exist create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    #clearing output path
    for filename in os.listdir(output_directory) :
        os.remove(os.path.join(output_directory, filename))

    with open(all_fields_lincolnbeet_JSON_file, "r") as f:
        data = json.load(f)
        data = data["annotations"]

    count = 0

    bounding_box_df = pd.read_csv(bounding_box_characterist_CSV_file)

    previous_filename = ""
    returned_mask_list = []
    for index, row in bounding_box_df.iterrows():

        #test wether or not the image exists
        output_name = f"{row.image_name}"
        output_path = os.path.join(output_directory, output_name)

        if os.path.exists(output_path):
            #the mask does exist let's use the existing mask as a source image
            mask = Image.open(output_path)
        else:
            #the mask does not exist let's create it out of the original image witdth and height

            #getting size of image to reproduce in mask
            img_input_name = row.image_name
            img_input_path = os.path.join(input_directory, img_input_name)
            if not os.path.exists(img_input_path):
                continue #file not in the target dataset
            img = Image.open(img_input_path)
            mask = create_empty_mask(img.width, img.height, output_path)

        #look for the bounding box of the image in the JSON file
        annotation = get_image_bbox_annotation(data, row.id)
        if annotation != None:
            image_id = annotation["image_id"]
            bbox = annotation["bbox"]  # [x_min, y_min, largeur, hauteur]
            category_id = annotation["category_id"]
            bbox_id = annotation["id"]

            x, y, w, h = bbox
            x = int(x); y = int(y) ;w = int(w); h = int(h)

            #use the bounding box as a white mask on top of the existing image
            left = x
            top = y
            right = left + w
            bottom = top + h

            bbbox_mask = Image.new('RGB', (w,h), (255,255,255))

            mask.paste(bbbox_mask, (x, y))
            mask.save(output_path)

        if row.image_name != previous_filename:
            previous_filename = row.image_name
            returned_mask_list.append(output_path)
            count += 1

        if max_image_generated!= None and count >= max_image_generated:
            print('maximum number of mask generated')
            break

    #check that all file in the input directory are represented with a mask
    for input_filename in os.listdir(input_directory) :
        output_filename = os.path.join(output_directory, os.path.basename(input_filename))
        if not os.path.exists(output_filename):
            img = Image.open(os.path.join(input_directory, input_filename))
            create_empty_mask(img.width, img.height, output_filename)

    return returned_mask_list
