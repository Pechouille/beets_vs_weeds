from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
import numpy as np
import os


def load_bounding_boxes(xml_file):
    '''Load bounding box object from the specified XML file'''

    tree = ET.parse(xml_file)
    root = tree.getroot()

    class bounding_box:
        def __init__(self, name, x_min, x_max, y_min, y_max):
            self.name = name
            self.x_min = float(x_min)
            self.x_max = float(x_max)
            self.y_min = float(y_min)
            self.y_max = float(y_max)
            self.x_len = self.x_max - self.x_min
            self.y_len = self.y_max - self.y_min
            if self.name == "sugar_beet":
                self.color = 'g'
            else:
                self.color = 'r'

    bound_boxes = []

    for object in root.iter('object'):
        bound_boxes.append(bounding_box(object.find('name').text,
                            object.find('bndbox/xmin').text,
                            object.find('bndbox/xmax').text,
                            object.find('bndbox/ymin').text,
                            object.find('bndbox/ymax').text)
                        )
    return bound_boxes

def display_image_with_bounding_boxes(image_path):
    '''Display the specified image with its attached bounding boxes
    both image filename and the XML file specifying the bounding boxes need to be in the same directory'''

    fig, axs = plt.subplots(1, 1, figsize=(20, 15))

    img = np.asarray(Image.open(image_path))
    # Create a Rectangle patch
    axs.imshow(img)

    xml_file_path = image_path[0:-3] + "xml"
    bounding_boxes = load_bounding_boxes(xml_file_path)

    for box in bounding_boxes:
        rect = patches.Rectangle((box.x_min, box.y_min), box.x_len, box.y_len, linewidth=2, edgecolor=box.color, facecolor=box.color, alpha = 0.6)
        axs.add_patch(rect)

    plt.show()

def api_display_image_with_bounding_boxes(image_path, save_path):
    '''Crée une image avec les bounding boxes et la sauvegarde dans save_path'''

    fig, axs = plt.subplots(1, 1, figsize=(20, 15))
    img = np.asarray(Image.open(image_path))
    axs.imshow(img)

    # Le fichier XML correspondant est dans data/all/ avec la même base de nom
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    xml_file_path = f"data/all/{base_name}.xml"

    bounding_boxes = load_bounding_boxes(xml_file_path)

    for box in bounding_boxes:
        rect = patches.Rectangle(
            (box.x_min, box.y_min), box.x_len, box.y_len,
            linewidth=2, edgecolor=box.color, facecolor=box.color, alpha=0.3
        )
        axs.add_patch(rect)

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
