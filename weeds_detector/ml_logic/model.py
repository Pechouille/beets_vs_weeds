from tensorflow.keras import Sequential, Input, layers, callbacks
import numpy as np

def initialize_model():

    """Initialize the Neural Network with random weights"""
    model = Sequential()
    model.add(Input(shape=(128, 128, 3)))

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))


    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    ### Third Convolution & MaxPooling
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    ### Fourth Convolution & MaxPooling
    model.add(layers.Conv2D(64, kernel_size=(2, 2), activation='relu'))

    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(10, activation='relu'))

    ### Last layer - Classification Layer with 2 outputs corresponding to circle and triangle
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def compile_model(model):
    model.compile(loss = 'binary_crossentropy', metrics = ['precision'], optimizer = 'adam')
    return model

def train_model(model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=32,
        patience=20,
        validation_data=None,
        validation_split=0.3):

    model = initialize_model()
    es = callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    model = compile_model(model)
    history = model.fit(X, y,
                        epochs = 100,
                        batch_size = batch_size,
                        validation_data = validation_data,
                        validation_split = validation_split,
                        callbacks=[es])
    return model, history
