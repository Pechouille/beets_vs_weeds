.DEFAULT_GOAL := default

##################### BUCKET ACTIONS #####################
reset_gcs_preprocessed_files:
	-gcloud storage folders delete gs://${BUCKET_NAME}/data/preprocessed
	-gcloud storage folders create --recursive gs://${BUCKET_NAME}/data/preprocessed

reinstall_package:
	@pip uninstall -y weeds_detector || :
	@pip install -e .


hello_world:
	-@echo 'Hello, this make section for Beets Vs Weeds works OK'

test_api:
	uvicorn api.fast:app --reload --host 0.0.0.0 --port 8000
