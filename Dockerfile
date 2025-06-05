# Use an official Python runtime as a parent image
FROM python:3.12.9

# Set the working directory in the container
WORKDIR /weeds_detector

# Copy the current directory contents into the container
COPY weeds_detector /weeds_detector
COPY requirements.txt /weeds_detector/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /weeds_detector/requirements.txt
RUN apt-get update && apt-get install -y make

# Make sure the command runs in the correct directory
WORKDIR /weeds_detector

# Run make hello_world when the container launches
CMD ["make", "hello_world"]
