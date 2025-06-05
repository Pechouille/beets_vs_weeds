import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")
FILE_ORIGIN = os.environ.get("FILE_ORIGIN")
LOCAL_DATA_PATH = os.environ.get("LOCAL_DATA_PATH")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
RESIZED = os.environ.get("RESIZED")
MODEL_TARGET = os.environ.get("MODEL_TARGET")

################## VALIDATIONS #################

env_valid_options = dict(
    FILE_ORIGIN=["local", "gcp"]
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)

##################  CONSTANTS  #####################
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".code", "Pechouille", "beets_vs_weeds", "training_outputs")
LOCAL_REGISTRY_PATH =os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "training_outputs")
