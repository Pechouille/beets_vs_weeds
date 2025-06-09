from tensorflow.keras import layers, models, Input, callbacks
from weeds_detector.params import *

def initialize_model(max_boxes=10, num_classes=1):
    resized = int(RESIZED)
    inputs = Input(shape=(resized, resized, 3))

    x = layers.Conv2D(16, kernel_size=(4, 4), activation='relu')(inputs)
    #x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)

    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)

    x = layers.Conv2D(64, kernel_size=(2, 2), activation='relu')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(resized, activation='relu')(x)

    # Sortie classification : max_boxes x num_classes
    class_output = layers.Dense(max_boxes * num_classes, activation='sigmoid')(x)
    class_output = layers.Reshape((max_boxes, num_classes), name='class_output')(class_output)

    # Sortie bounding boxes : max_boxes x 4 coordonnées
    bbox_output = layers.Dense(max_boxes * 4, activation='sigmoid')(x)  # sigmoid: pour normaliser entre 0 et 1
    bbox_output = layers.Reshape((max_boxes, 4), name='bbox_output')(bbox_output)

    model = models.Model(inputs=inputs, outputs=[class_output, bbox_output])
    return model

def compile_model(model):

    model.compile(
        loss = {
            'class_output': 'binary_crossentropy',
            'bbox_output': 'mean_squared_error'
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
        batch_size=32,
        patience=20,
        epochs = 100,
        validation_split=0.3):

        es = callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

        history = model.fit(X,
                    {'class_output': y_class,'bbox_output': y_bbox},
                    epochs = epochs,
                    batch_size = batch_size,
                    validation_split = validation_split,
                    callbacks=[es])

        return model, history
