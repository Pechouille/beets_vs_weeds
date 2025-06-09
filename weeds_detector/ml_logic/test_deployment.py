from weeds_detector.ml_logic.preprocess_model_segm_class import *
from weeds_detector.ml_logic.training_model_segm_class import *
from weeds_detector.ml_logic.registry import *
X_prepro = preprocess_images(10)
y_bbox, y_class = preprocess_y(10)
model = initialize_model(max_boxes=10, num_classes=1)
model = compile_model(model)
model, history = train_model(model,
        X_prepro,
        y_class,
        y_bbox,
        batch_size=32,
        patience=50,
        epochs = 5000,
        validation_data=None,
        validation_split=0.3)
save_model(model)
