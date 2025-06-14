from tensorflow.keras import layers, models, Input, callbacks
from weeds_detector.params import *

import tensorflow.keras.backend as K

def masked_mse(y_true, y_pred):
    # Mask: 1 if bbox ≠ [0,0,0,0], else 0
    mask = K.cast(K.any(y_true != 0.0, axis=-1), dtype='float32')  # shape: (batch, max_boxes)
    mask = K.expand_dims(mask, axis=-1)  # shape: (batch, max_boxes, 1)
    mask = K.repeat_elements(mask, 4, axis=-1)  # shape: (batch, max_boxes, 4)

    # Apply mask to error
    squared_error = K.square(y_true - y_pred)
    masked_se = squared_error * mask

    return K.sum(masked_se) / (K.sum(mask) + K.epsilon())


def initialize_model(max_boxes=5, num_classes=1):
    resized = int(RESIZED)
    inputs = Input(shape=(resized, resized, 3))

    x = layers.Conv2D(16, kernel_size=(4, 4), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)


    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)

    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(x)

    x = layers.GlobalAveragePooling2D()(x)  # Remplace Flatten

    x = layers.Dense(256, activation='relu')(x)  # Petit Dense suffisant

    # Sortie classification : max_boxes x num_classes
    class_output = layers.Dense(max_boxes * num_classes, activation='sigmoid')(x)
    class_output = layers.Reshape((max_boxes, num_classes), name='class_output')(class_output)

    # Sortie bounding boxes : max_boxes x 4 coordonnées
    bbox_output = layers.Dense(max_boxes * 4, activation='sigmoid')(x)
    bbox_output = layers.Reshape((max_boxes, 4), name='bbox_output')(bbox_output)

    model = models.Model(inputs=inputs, outputs=[class_output, bbox_output])
    return model


def compile_model(model):
    model.compile(
        loss = {
            'class_output': 'binary_crossentropy',
            'bbox_output': masked_mse
        },
        metrics = {
            'class_output': ['precision'],
            'bbox_output': ['mean_absolute_error']
        },
        optimizer = 'adam'
    )

    return model

def train_model(model,
                X,
                y_class,
                y_bbox,
                batch_size=7,
                patience=20,
                epochs=100,
                validation_split=0.3):

    es = callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

    # Cibles à prédire
    y_targets = {
        'class_output': y_class,
        'bbox_output': y_bbox
    }

    print("y_class shape:", y_class.shape)
    print("y_bbox shape:", y_bbox.shape)

    history = model.fit(
        X,
        y_targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[es])

    return model, history
