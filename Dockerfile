# Use an official TensorFlow runtime as a parent image (Python 3.11, CPU version)
FROM tensorflow/tensorflow:2.16.1

# Set the working directory in the container

RUN mkdir /app
WORKDIR /app

ADD . .

# Install FastAPI and Uvicorn, as TensorFlow image already includes TensorFlow
RUN pip install --no-cache-dir fastapi[all] uvicorn python-multipart Pillow

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run app.py when the container launches using Uvicorn
CMD ["uvicorn", "src.api.v1.main:app", "--host", "0.0.0.0", "--port", "8000"]
