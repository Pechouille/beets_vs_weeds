from weeds_detector.ml_logic.preprocess_model_segm_class import *
from weeds_detector.ml_logic.training_model_segm_class import *
from weeds_detector.ml_logic.registry import *


X_prepro, filenames_ordered = preprocess_images(5)
y_bbox, y_class = preprocess_y(filenames_ordered, 5)
print('Preprocessed of X and y done')

model = initialize_model(max_boxes=5, num_classes=1)
print('Model initialized')

model = compile_model(model)
print('Model compiled')


model, history = train_model(
    model,
    X_prepro,
    y_class,
    y_bbox,
    batch_size=128,
    patience=2,
    epochs=2,
    validation_split=0.3
)
print('Model trained')

save_model(model, 'cnn_segm_classif_2')
