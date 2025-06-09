from weeds_detector.ml_logic.preprocess_model_class import preprocess_features
from weeds_detector.ml_logic.training_model_class import *
from weeds_detector.ml_logic.registry import *


X_prepro, y = preprocess_features()

model = initialize_model()
model = compile_model(model)

model, history = train_model(model,
        X_prepro,
        y,
        batch_size=32,
        epochs=100,
        patience=20,
        validation_data=None,
        validation_split=0.3)

save_model(model)
