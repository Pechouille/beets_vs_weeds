.DEFAULT_GOAL := default

##################### AUTHENTICATION #####################
auth_gcp:
	@echo "Authenticating with service account..."
	@if [ -f "/tmp/service-account.json" ]; then \
		gcloud auth activate-service-account --key-file=/tmp/service-account.json; \
		gcloud config set project $(PROJECT_ID); \
	else \
		echo "Service account key not found. Using default authentication."; \
	fi

##################### BUCKET ACTIONS #####################
reset_gcs_preprocessed_files: auth_gcp
	@echo "Resetting GCS preprocessed files..."
	-gcloud storage rm gs://$(BUCKET_NAME)/data/preprocessed/** --recursive
	-gcloud storage folders create gs://$(BUCKET_NAME)/data/preprocessed

list_gcs_files: auth_gcp
	@echo "Listing files in bucket..."
	gcloud storage ls gs://$(BUCKET_NAME)/data/ --recursive

##################### PACKAGE ACTIONS #####################
reinstall_package:
	@echo "Reinstalling package..."
	@pip uninstall -y weeds_detector || :
	@pip install -e .

##################### TESTING #####################
hello_world:
	@echo 'Hello, this make section for Beets Vs Weeds works OK'
	@echo "Docker container is running in: $(PWD)"
	@echo "Available make targets:"
	@echo "  - hello_world: This message"
	@echo "  - reset_gcs_preprocessed_files: Reset GCS files"
	@echo "  - list_gcs_files: List GCS files"
	@echo "  - reinstall_package: Reinstall Python package"

##################### DEFAULT #####################
default: hello_world
