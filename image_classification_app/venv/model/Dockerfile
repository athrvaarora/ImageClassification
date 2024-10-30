# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:2.11.0

# Set the working directory in the container
WORKDIR /app

# Copy the model and server script into the container
COPY model/ /app/model/
COPY model_server.py /app/model_server.py

# Install any additional Python dependencies
RUN pip install flask

# Expose the port your application runs on
EXPOSE 8080

# Command to run the server
CMD ["python", "model_server.py"]
