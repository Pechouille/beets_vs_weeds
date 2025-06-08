from tensorflow.keras import Sequential, Input, layers, callbacks, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import random
import glob


def dice_coeff(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

#Let's create a function for one step of the encoder block, so as to increase the reusability when making custom unets

def encoder_block(filters, inputs):
  x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(inputs)
  s = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
  p = MaxPooling2D(pool_size = (2,2), padding = 'same')(s)
  return s, p #p provides the input to the next encoder block and s provides the context/features to the symmetrically opposte decoder block

#Baseline layer is just a bunch on Convolutional Layers to extract high level features from the downsampled Image
def baseline_layer(filters, inputs):
  x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(inputs)
  x = Conv2D(filters, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(x)
  return x

#Decoder Block
def decoder_block(filters, connections, inputs):
  x = Conv2DTranspose(filters, kernel_size = (2,2), padding = 'same', activation = 'relu', strides = 2)(inputs)
  skip_connections = concatenate([x, connections], axis = -1)
  x = Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = 'relu')(skip_connections)
  x = Conv2D(filters, kernel_size = (2,2), padding = 'same', activation = 'relu')(x)
  return x




def initialize_model():

    """Initialize the Neural Network with random weights"""
    #Defining the input layer and specifying the shape of the images
    inputs = Input(shape = (256,256, 3))

    #defining the encoder
    s1, p1 = encoder_block(64, inputs = inputs)
    s2, p2 = encoder_block(128, inputs = p1)
    s3, p3 = encoder_block(256, inputs = p2)
    s4, p4 = encoder_block(512, inputs = p3)

    #Setting up the baseline
    baseline = baseline_layer(1024, p4)

    #Defining the entire decoder
    d1 = decoder_block(512, s4, baseline)
    d2 = decoder_block(256, s3, d1)
    d3 = decoder_block(128, s2, d2)
    d4 = decoder_block(64, s1, d3)

    #Setting up the output function for binary classification of pixels
    outputs = Conv2D(1, 1, activation = 'sigmoid')(d4)

    #Finalizing the model
    model = Model(inputs = inputs, outputs = outputs, name = 'Unet')

    return model

def compile_model(model):

    model.compile(loss = 'binary_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy', dice_coeff])

    return model


def build_model_input_dataset(image_directory, mask_directory, image_size = 256, batch_size=8):


    # Get and obtain image and mask paths
    image_paths = sorted([os.path.join(image_directory, fname) for fname in os.listdir(image_directory)])
    mask_paths = sorted([os.path.join(mask_directory, fname) for fname in os.listdir(mask_directory)])

    def process_pair(img_path, mask_path):
        # Load RGB images
        image = load_img(img_path, target_size=image_size)
        image = img_to_array(image) / 255.0  # Normalize

        # Load mask a grayscale images
        mask = load_img(mask_path, target_size=image_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0    #normalize

        return image, mask

    # Dataset Generator
    def generator():
        for img, msk in zip(image_paths, mask_paths):
            yield process_pair(img, msk)

    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([image_size, image_size, 3]),
            tf.TensorShape([image_size, image_size, 1])
        )
    )

    # Optimize (batch, shuffle, prefetch)
    dataset = dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

def train_model(model,
        dataset,
        batch_size=64,
        patience=20,
        validation_data=None,
        validation_split=0.3):

    #Defining early stopping to regularize the model and prevent overfitting
    early_stopping = callbacks.EarlyStopping(monitor = 'val_loss', patience = patience, restore_best_weights = True)

    #Training the model with 50 epochs (it will stop training in between because of early stopping)
    history = model.fit(dataset, epochs = 1, batch_size = batch_size, callbacks = [early_stopping])
    return model, history

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
