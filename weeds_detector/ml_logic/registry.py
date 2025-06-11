import time
import os
from weeds_detector.params import LOCAL_REGISTRY_PATH, BUCKET_NAME, MODEL_TARGET
from tensorflow import keras
from google.cloud import storage
from colorama import Fore, Style
from glob import glob
import requests

def save_model(model: keras.Model, model_type: str) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{model_type}_{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{model_type}_{timestamp}.h5" --> unit 02 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{model_type}_{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":

        model_filename = model_path.split("/")[-1]
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)
        print("✅ Model saved to GCS")
        os.remove(model_path)
        print("🗑️ Local model file removed after upload to GCS")

        return None


def load_model(model_type: str):
        if MODEL_TARGET == "local":
            print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)
            # Get the latest model version name by the timestamp on disk
            local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
            local_model_paths = glob.glob(f"{local_model_directory}/{model_type}_*")

            if not local_model_paths:
                return None

            most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

            print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

            latest_model = keras.models.load_model(most_recent_model_path_on_disk)

            print("✅ Model loaded from local disk")

            return latest_model

        elif MODEL_TARGET == "gcs":

            print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

            client = storage.Client()
            blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix=f"models/{model_type}_"))

            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            os.makedirs(os.path.dirname(latest_model_path_to_save), exist_ok=True)
            latest_blob.download_to_filename(latest_model_path_to_save)
            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
