from weeds_detector.ml_logic.model_UNET import build_dataset, initialize_model, compile_model, train_model
from weeds_detector.ml_logic.registry import save_model

train_dataset = build_dataset(image_dir='images_preprocessed/UNET_images/train', mask_dir='images_preprocessed/UNET_masks/train')
val_dataset = build_dataset(image_dir='images_preprocessed/UNET_images/val', mask_dir='images_preprocessed/UNET_masks/val')

model = initialize_model()
model = compile_model(model)

model, history = train_model(
    model,
    train_dataset,
    batch_size=64,
    epochs=300,
    patience=20,
    validation_data=val_dataset
)

save_model(model, 'unet_segmentation_model')
