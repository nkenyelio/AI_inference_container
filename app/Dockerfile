# Use an official Python runtime as a parent image
FROM python:3.8.0-slim
# Set the working directory to /app

# Install prerequires

RUN apt-get update && apt-get install -y curl

WORKDIR /app
# Copy the current directory contents into the container at /app

COPY . /app
# Install any needed packages specified in requirements.txt

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# Make port 8000 available to the world outside this container

#EXPOSE 5000
# Run run_keras_server.py when the container launches

CMD ["python3", "run_keras_server.py"]
#CMD ["python", "simple_request.py"]