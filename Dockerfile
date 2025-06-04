FROM python:3.12.9

COPY weeds_detector /weeds_detector
COPY dev_requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT
