import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
RESIZED = os.environ.get("RESIZED")
