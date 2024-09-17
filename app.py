from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import boto3
import os
import json
from fastapi import FastAPI
from contextlib import asynccontextmanager

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_DEFAULT_REGION')
AWS_BUCKET = os.getenv('AWS_BUCKET', 'plantpulse')

s3_client = boto3.client('s3', region_name=AWS_REGION, aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

MODEL_PATH = "models/best_model.keras"

def download_model_from_s3(bucket_name, s3_key, local_path):
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        print(f"Model downloaded successfully from S3: {s3_key}")
    except Exception as e:
        print(f"Error downloading model from S3: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs before the app starts
    s3_key = "best_model.keras"
    download_model_from_s3(AWS_BUCKET, s3_key, MODEL_PATH)

    global model
    model = tf.keras.models.load_model(MODEL_PATH)

    global class_labels
    with open('classes.json', 'r') as f:
        class_labels = json.load(f)

    yield  # This yields control back to the application during its lifetime

    print("Shutting down application...")

app = FastAPI(lifespan=lifespan)

def prepare_image(img, target_size=(224, 224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image to [0,1]
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    img = Image.open(BytesIO(image_data))

    processed_image = prepare_image(img)

    predictions = model.predict(processed_image)
    print("Raw predictions:", predictions)

    predicted_class_idx = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    predicted_class = class_labels[str(predicted_class_idx)]

    return {
        "predicted_class_index": predicted_class_idx,
        "predicted_class": predicted_class,
        "confidence": confidence
    }
