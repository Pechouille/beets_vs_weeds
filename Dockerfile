FROM python:3.12.9

COPY weeds_detector /weeds_detector
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y make

CMD make hello_world
