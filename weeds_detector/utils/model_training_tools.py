import shutil
import json
import os


def build_training_directory(pattern, definition_directory, source_directory, destination_directory, max_number_of_files = None):
    ''' Build train, test or val directory from the corresponding definition file found in the <definition directory>
    into the <destination_directory>. Can limit the number of file copy/paste through the <max_number_of_file>
    Example:
        - train_files = build_training_directory("train", "./data/all_fields_lincolnbeet/", "data", "data", 80)
        build the the "train" dataset out of the file "./data/all_fields_lincolnbeet/all_fields_lincolnbeet_train_.json"
        file.
        Source image file will be extracted from the "data" directory
        and copied into the data/train directory
        the train dataset will be built out of first 80 files found
    '''
    with open(os.path.join(definition_directory, f"all_fields_lincolnbeet_{pattern}_.json"), "r") as f:
        data = json.load(f)

    destination = os.path.join(destination_directory, pattern)

    if not os.path.exists(destination):
        os.makedirs(destination)

    for filename in os.listdir(destination) :
        os.remove(os.path.join(destination, filename))

    nb_files = 0
    dataset = []
    for file in data:
        source_file = os.path.join(source_directory, file)
        destination_file = os.path.join(destination, os.path.basename(file))
        shutil.copyfile(source_file, destination_file)
        nb_files+=1
        dataset.append(destination_file)
        if max_number_of_files != None and nb_files >= max_number_of_files:
            break;

    return dataset
