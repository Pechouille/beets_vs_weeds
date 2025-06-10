# Use an official Python runtime as a parent image
FROM python:3.12.9
COPY weeds_detector /weeds_detector
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn weeds_detector.api.fast:app --host 0.0.0.0 --port $PORT
