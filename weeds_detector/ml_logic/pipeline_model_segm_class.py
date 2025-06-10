from weeds_detector.ml_logic.preprocess_model_segm_class import *
from weeds_detector.ml_logic.training_model_segm_class import *
from weeds_detector.ml_logic.registry import *


X_prepro, filenames_ordered = preprocess_images(2)
y_bbox, y_class, mask = preprocess_y(filenames_ordered, 2)

model = initialize_model(max_boxes=2, num_classes=1)
model = compile_model(model)

model, history = train_model(
    model,
    X_prepro,
    y_class,
    y_bbox,
    batch_size=7,
    patience=2,
    epochs=10,
    validation_split=0.3
)

save_model(model)
