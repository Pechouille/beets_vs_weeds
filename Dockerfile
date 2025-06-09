# Use an official Python runtime as a parent image
FROM python:3.12.9

# Set the working directory in the container
WORKDIR /weeds_detector

# Copy the current directory contents into the container
COPY weeds_detector /weeds_detector
COPY requirements.txt /requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

# Make sure the command runs in the correct directory
WORKDIR /weeds_detector

# Default command - can be overridden
CMD ["make", "hello_world"]
