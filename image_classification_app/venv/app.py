from flask import Flask, request, render_template, jsonify
from google.cloud import storage, bigquery
import requests
import os

app = Flask(__name__)

PROJECT_ID = "imageclassification-440205"   # Replace with your GCP project ID
BUCKET_NAME = "imageclassificationwebapp" # Replace with your Cloud Storage bucket name
DATASET_ID = "image_classification"   # Replace with your BigQuery dataset ID
TABLE_ID = "classification_results"       # Replace with your BigQuery table ID
ML_MODEL_URL = "http://34.68.92.244/predict" # Replace with your Kubernetes model service URL

# Initialize Google Cloud clients
storage_client = storage.Client()
bigquery_client = bigquery.Client()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save file temporarily
    filepath = os.path.join("/tmp", file.filename)
    file.save(filepath)

    # Upload to Cloud Storage
    upload_to_storage(BUCKET_NAME, filepath, file.filename)

    # Send image to ML model
    with open(filepath, "rb") as image:
        response = requests.post(ML_MODEL_URL, files={"file": image})
        if response.status_code != 200:
            return jsonify({"error": "Failed to classify image"}), 500
        classification_result = response.json()

    # Save classification result in BigQuery
    save_to_bigquery(DATASET_ID, TABLE_ID, classification_result)

    # Clean up temporary file
    os.remove(filepath)

    return jsonify({
        "message": "File uploaded and classified successfully!",
        "classification_result": classification_result
    })

def upload_to_storage(bucket_name, source_file_name, destination_blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def save_to_bigquery(dataset_id, table_id, classification_result):
    table_ref = bigquery_client.dataset(dataset_id).table(table_id)
    rows_to_insert = [classification_result]
    bigquery_client.insert_rows_json(table_ref, rows_to_insert)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
