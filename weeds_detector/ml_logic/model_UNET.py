import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras import Input, callbacks, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import get_file
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.image import resize, decode_png
from tensorflow.io import read_file
from tensorflow import Tensor, cast, clip_by_value, reduce_sum, reduce_mean, float32, uint8, constant
from typing import Tuple
from weeds_detector.data import get_all_files_path_and_name_in_directory, get_content_from_url

from weeds_detector.params import RESIZED, FILE_ORIGIN

def dice_coeff(y_true: Tensor, y_pred: Tensor, smooth: float = 1e-6) -> float:
    """
    Computes the Dice coefficient, a measure of overlap between two samples.

    Parameters:
    y_true (tf.Tensor): The ground truth labels.
    y_pred (tf.Tensor): The predicted labels.
    smooth (float): A small constant added to the numerator and denominator to avoid division by zero.

    Returns:
    float: The mean Dice coefficient over the batch.
    """
    y_true = cast(y_true, float32)  # Conversion obligatoire
    y_pred = clip_by_value(y_pred, 0., 1.)
    intersection = reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = reduce_sum(y_true, axis=[1,2,3]) + reduce_sum(y_pred, axis=[1,2,3])
    dice = (2. * intersection + smooth) / (union + smooth)

    return reduce_mean(dice)

def dice_loss(y_true: Tensor, y_pred: Tensor, smooth: float = 1e-6) -> Tensor:
    """
    Calculate the Dice Loss between the true and predicted tensors.

    The Dice Loss is a measure of the overlap between two sets, and is often used
    in image segmentation tasks. It is defined as 1 minus the Dice Coefficient.

    Parameters:
    y_true (tf.Tensor): The ground truth tensor.
    y_pred (tf.Tensor): The predicted tensor.
    smooth (float): A small constant added to the numerator and denominator to
                    prevent division by zero.

    Returns:
        Tensor: The Dice Loss.
    """
    y_true = cast(y_true, float32)
    y_pred = clip_by_value(y_pred, 0., 1.)

    intersection = reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = reduce_sum(y_true, axis=[1, 2, 3]) + reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - reduce_mean(dice)

def combined_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculate the combined loss, which is the average of Binary Crossentropy and Dice Loss.

    This function combines two loss functions: Binary Crossentropy and Dice Loss. The combined loss
    is the average of these two losses, providing a balanced measure of model performance.

    Parameters:
    y_true (tf.Tensor): The ground truth tensor.
    y_pred (tf.Tensor): The predicted tensor.

    Returns:
    tf.Tensor: The combined loss.
    """
    bce = BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

#Let's create a function for one step of the encoder block, so as to increase the reusability when making custom unets
def encoder_block(filters: int, inputs: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Encoder block for the U-Net model.

    This block consists of two convolutional layers followed by a max pooling layer.

    Args:
        filters (int): Number of filters for the convolutional layers.
        inputs (tf.Tensor): Input tensor to the encoder block.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the output tensor after the second convolutional layer
                                      and the output tensor after the max pooling layer.
    """
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(inputs)
    s = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(x)
    p = MaxPooling2D(pool_size=(2, 2), padding='same')(s)
    return s, p #p provides the input to the next encoder block and s provides the context/features to the symmetrically opposte decoder block

#Baseline layer is just a bunch on Convolutional Layers to extract high level features from the downsampled Image
def baseline_layer(filters: int, inputs: Tensor) -> Tensor:
    """
    Baseline layer for the U-Net model.

    This layer consists of two convolutional layers to extract high-level features from the downsampled image.

    Args:
        filters (int): Number of filters for the convolutional layers.
        inputs (Tensor): Input tensor to the baseline layer.

    Returns:
        Tensor: Output tensor after the second convolutional layer.
    """
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(inputs)
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(x)
    return x

#Decoder Block
def decoder_block(filters: int, connections: Tensor, inputs: Tensor) -> Tensor:
    """
    Decoder block for the U-Net model.

    This block consists of a transposed convolutional layer followed by concatenation with skip connections,
    and then two convolutional layers.

    Args:
        filters (int): Number of filters for the convolutional layers.
        connections (Tensor): Skip connections tensor to be concatenated.
        inputs (Tensor): Input tensor to the decoder block.

    Returns:
        Tensor: Output tensor after the second convolutional layer.
    """
    x = Conv2DTranspose(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=2)(inputs)
    skip_connections = concatenate([x, connections], axis=-1)
    x = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu')(skip_connections)
    x = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu')(x)
    return x

def initialize_model() -> Model:
    """
    Initialize the Neural Network with random weights.

    This function defines the U-Net model architecture, including the encoder, baseline, and decoder blocks.

    Returns:
        Model: The initialized U-Net model.
    """
    # Defining the input layer and specifying the shape of the images
    resized = int(RESIZED)
    inputs = Input(shape=(resized, resized, 3))

    # Defining the encoder
    s1, p1 = encoder_block(resized // 4, inputs=inputs)
    s2, p2 = encoder_block(resized // 2, inputs=p1)
    s3, p3 = encoder_block(resized, inputs=p2)
    s4, p4 = encoder_block(resized * 2, inputs=p3)

    # Setting up the baseline
    baseline = baseline_layer(resized*4, p4)

    # Defining the entire decoder
    d1 = decoder_block(resized * 2, s4, baseline)
    d2 = decoder_block(resized, s3, d1)
    d3 = decoder_block(resized // 2, s2, d2)
    d4 = decoder_block(resized // 4, s1, d3)

    # Setting up the output function for binary classification of pixels
    outputs = Conv2D(1, 1, activation='sigmoid')(d4)

    # Finalizing the model
    model = Model(inputs=inputs, outputs=outputs, name='Unet')
    return model

def compile_model(model: Model) -> Model:
    """
    Compile the model.

    Args:
        model (Model): The U-Net model to be compiled.

    Returns:
        Model: The compiled U-Net model.
    """
    model.compile(optimizer='adam',
                  loss=combined_loss,
                  metrics=[dice_coeff])
    return model

def process_path(image_path: str, mask_path: str):
    '''This methode is only used in load_dataset and shall not be used anywhere else
    it automaically normalize the input image before inserting them in the dataset'''
    # Chargement du fichier image
    print(f"Process image : {image_path} and mask : {mask_path}")
    image_size = int(RESIZED)

    if FILE_ORIGIN == 'gcp':
        image = get_content_from_url(image_path)
        mask = get_content_from_url(mask_path)
    else:
        image = read_file(image_path)
        mask = read_file(mask_path)

    print("Decode image PNG")
    image = decode_png(image, channels=3)  # couleur
    image = resize(image, [image_size, image_size])
    image = cast(image, float32) / 255.0  # normalisation [0, 1]

    print("Decode mask PNG")
    mask = decode_png(mask, channels=1)  # niveau de gris (binaire ou multiclasses)
    mask = resize(mask, [image_size, image_size], method='nearest')  # nearest pour préserver les classes
    mask = cast(mask, float32) / 255.0  # typiquement les masques sont des entiers (classe 0, 1, 2…)

    return image, mask

def pair_files_image_mask(image_paths, mask_paths):
    """Pair each image with its corresponding mask using filenames."""
    mask_dict = { mask[1]: mask[0] for mask in mask_paths}
    pair_urls = []
    counter = 1
    for image_url, image_filename in image_paths:
        if image_filename in mask_dict:
                pair_urls.append([image_url, mask_dict[image_filename]])
                counter += 1

    return pair_urls

def build_dataset(image_dir='images_preprocessed/UNET_images/train', mask_dir='images_preprocessed/UNET_masks/train', batch_size=16):
    '''Build dataset which are consumed but the train process'''
    print("Start Build Dataset")
    print("Load files names from directories")
    image_paths = get_all_files_path_and_name_in_directory(image_dir, [".png"])
    mask_paths = get_all_files_path_and_name_in_directory(mask_dir, [".png"])
    print("Paired image_path and mask_path if mask exist")
    pair_urls = pair_files_image_mask(image_paths, mask_paths)
    image_list = []
    mask_list = []
    counter = 1
    nbr_images = len(pair_urls)
    print("Process each images")
    print(f"Number of image to process : { nbr_images }")
    for image_url, mask_url in pair_urls:
        print(f"{counter} / {nbr_images} : Process Image : {image_url} and Mask : {mask_url}")
        image, mask = process_path(image_url, mask_url)
        image_list.append(image)
        mask_list.append(mask)
        counter += 1

    print("Finish process images")
    print("Define Tensorflow Dataset")
    dataset = Dataset.from_tensor_slices((image_list, mask_list))
    print("Shuffle dataset")
    dataset = dataset.shuffle(buffer_size=100)
    print("Build batch")
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def train_model(model: Model,
                dataset: Dataset,
                batch_size: int = 32,
                patience: int = 20,
                epochs: int = 50,
                validation_data: Dataset = None,
                validation_split: float = 0.3) -> Tuple[Model, dict]:
    """
    Train the model.

    Args:
        model (Model): The U-Net model to be trained.
        dataset (Dataset): The training dataset.
        batch_size (int, optional): The batch size for training. Defaults to 32.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 20.
        epochs (int, optional): The number of epochs to train the model. Defaults to 50.
        validation_data (Dataset, optional): The validation dataset. Defaults to None.
        validation_split (float, optional): The fraction of the training data to be used as validation data. Defaults to 0.3.

    Returns:
        Tuple[Model, dict]: The trained model and the training history.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(dataset,
                        validation_data=validation_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[early_stopping])
    return model, history

def plot_history(history):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history.get('val_loss'), label='val_loss')
    plt.legend()
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(history.history['dice_coeff'], label='dice')
    plt.plot(history.history.get('val_dice_coeff'), label='val_dice')
    plt.legend()
    plt.title("Dice Coefficient")

    plt.show()

def evaluate_model(
        model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64):
    """
    Evaluate trained model performance on the dataset
    """

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    print(X.shape, y.shape)
    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    return metrics

def predict(model: Model, image_path: str, image_size: int):
    """
    Predicts the mask for a given image using the provided model.

    Args:
        model (Model): The trained model used for prediction.
        image_path (str): The file path to the input image.
        image_size (int): The size to which the image should be resized.

    Returns:
        np.ndarray: The predicted mask for the input image.
    """

    image = read_file(image_path)
    image = decode_png(image, channels=3)  # couleur
    image = resize(image, [image_size, image_size])
    image = cast(image, float32) / 255.0  # normalisation [0, 1]
    return model.predict(image)
