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

def generate_custom_bounding_box_image(bounding_box_characterist_CSV_file,
                                        all_fields_lincolnbeet_JSON_file,
                                        input_directory,
                                        output_directory,
                                        new_bounding_size,
                                        output_size, max_image_generated = None):
    '''Rebuild the bounding box located in <input_directory> based on the bounding box described in
    <bounding_box_characterist_CSV_file> and <all_fields_lincolnbeet_JSON_file>
    filter from the image description file only bounding box which h x w are inferior to new_bounding_size
    build a new squared bounding box out of this new size
    resize the image to a squared image of output size'''

    #clearing output path
    for filename in os.listdir(output_directory) :
        os.remove(os.path.join(output_directory, filename))

    with open(all_fields_lincolnbeet_JSON_file, "r") as f:
        data = json.load(f)
        data = data["annotations"]

    count = 0

    bounding_box_df = pd.read_csv(bounding_box_characterist_CSV_file)
    bounding_box_df = bounding_box_df[bounding_box_df.overlapping_degree == 0][bounding_box_df.width < new_bounding_size][bounding_box_df.width < new_bounding_size]

    for index, row in bounding_box_df.iterrows():
        #look for the bounding box of the image in the JSON file

        annotation = get_image_bbox_annotation(data, row.id)
        if annotation == None:
            raise Exception(f"generate_bounding_box_image: Image {row.id} - {row.image_name} not found in annotation file: {all_fields_lincolnbeet_JSON_file}")

        image_id = annotation["image_id"]
        bbox = annotation["bbox"]  # [x_min, y_min, largeur, hauteur]
        category_id = annotation["category_id"]
        bbox_id = annotation["id"]

        image = Image.open(os.path.join(input_directory, row.image_name))

        x, y, w, h = bbox
        x = int(x); y = int(y) ;w = int(w); h = int(h)

        #Get center of bounding box then set the x/y and set the bounding box to be bounding x bounding size
        left = x + w/2 - new_bounding_size/2
        top = y + h/2 - new_bounding_size/2
        right = left + new_bounding_size
        bottom = top + new_bounding_size

        if left < 0:
            translate = left * -1
            left = 0
            right += translate

        if top < 0:
            translate = top * -1
            top = 0
            bottom += translate

        if right > image.size[0]:
            translate = right - image.size[0]
            right = image.size[0]
            left -= translate

        if bottom > image.size[1]:
            translate = bottom - image.size[1]
            bottom = image.size[1]
            top -= translate

        #crop image to bounding size
        cropped = image.crop((left, top, right, bottom))

        #resize image to reduce resolution
        resized = cropped.resize((output_size,output_size))

        output_name = f"{row.image_name}_{image_id}_{bbox_id}_{category_id}.png"
        output_path = os.path.join(output_directory, output_name)

        resized.save(output_path)
        count += 1

        if max_image_generated!= None and count >= max_image_generated:
            print('maximum number of image generated')
            break
