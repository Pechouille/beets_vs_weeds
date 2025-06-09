from tensorflow.keras import Sequential, Input, layers, callbacks, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

import random
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Setting dice coefficient to evaluate our model
def dice_coeff(y_true, y_pred, smooth = 1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true*y_pred, axis = -1)
    union = tf.reduce_sum(y_true, axis = -1) + tf.reduce_sum(y_pred, axis = -1)
    dice_coeff = (2*intersection+smooth) / (union + smooth)
    return dice_coeff

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

def process_path(image_path, mask_path):
    '''This methode is only used in load_dataset and shall not be used anywhere else
    it automaically normalize the input image before inserting them in the dataset'''
    # Chargement du fichier image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # couleur
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32) / 255.0  # normalisation [0, 1]

    # Chargement du masque
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)  # niveau de gris (binaire ou multiclasses)
    mask = tf.image.resize(mask, [256, 256], method='nearest')  # nearest pour préserver les classes
    mask = tf.cast(mask, tf.uint8)  # typiquement les masques sont des entiers (classe 0, 1, 2…)

    return image, mask

def build_dataset(image_dir, mask_dir, batch_size=16):
    '''Build dataset which are consumed but the train process'''
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
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


def predict(model, img):
    return model.predict(img)
