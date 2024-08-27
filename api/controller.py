import asyncio
import base64
import sys
import threading

import torch
import logging

import json

from api.utils import make_deeplab
import joblib

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io
import gzip

from api.model_predictions import predict
from api.metrics_calculation import calculate_final_metrics
from api.open_ai import generate_health_report
from api.request_model import PredictRequest

app = FastAPI()

# Load models and scaler once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deeplab = make_deeplab(device)

model_wrist = joblib.load("models/wrist_ensemble_model.pkl")
model_waist = joblib.load("models/waist_ensemble_model.pkl")
model_hip = joblib.load("models/hip_ensemble_model.pkl")
scaler = joblib.load("models/scaler.pkl")

model_lock = threading.Lock()

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Format of the log messages
    handlers=[
        logging.StreamHandler(sys.stdout)  # Output logs to stdout (console)
    ]
)

logger = logging.getLogger("fastmind")


def compress_data(data):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(data.encode('utf-8'))
    return buf.getvalue()


async def process_image_async(image_data: bytes):
    # Simulate I/O-bound operation
    await asyncio.sleep(0.1)  # Example: simulate I/O delay

    # Open the image asynchronously
    image = Image.open(io.BytesIO(image_data))
    return image


@app.get("/")
async def health_check():
    return {"message": "The Health Check is successful"}


@app.post("/create-request/")
async def create_request(request: Request, height: float,
                         weight: float,
                         age: int,
                         gender: str,
                         file_front: UploadFile = File(...),
                         file_left: UploadFile = File(...)):

    # Convert the uploaded files to Base64
    front_encoded = (await convert_image_to_base64(file_front))
    left_encoded = (await convert_image_to_base64(file_left))

    # Build the request body
    body = {
        "height": height,
        "weight": weight,
        "age": age,
        "gender": gender,
        "file_front": front_encoded,
        "file_left": left_encoded
    }
    # Extract the request details
    method = "POST"
    url = str(request.url)

    # Generate the curl command
    curl_command = f"curl -X {method} '{url}'"

    # Add the body to the curl command
    curl_command += f" -H 'Content-Type: application/json' -d '{json.dumps(body)}'"

    return curl_command


@app.post("/convert-to-base64/")
async def convert_image_to_base64(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        file_data = await file.read()

        # Compress the file data using gzip
        compressed_data = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed_data, mode='wb') as gzip_file:
            gzip_file.write(file_data)

        # Move the pointer to the start of the buffer
        compressed_data.seek(0)

        # Encode the compressed data to Base64
        base64_encoded = base64.b64encode(compressed_data.read()).decode('utf-8')

        # Return the Base64 encoded string
        return base64_encoded

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/")
async def predict_bfp_bmi_fmi(predict_request: PredictRequest):
    try:

        # Extract the relevant fields from the body
        height = predict_request.height
        weight = predict_request.weight
        age = predict_request.age
        gender = predict_request.gender

        # Decode Base64 strings
        compressed_file_front_data = base64.b64decode(predict_request.file_front)
        compressed_file_left_data = base64.b64decode(predict_request.file_left)

        # Decompress the data using gzip
        with gzip.GzipFile(fileobj=io.BytesIO(compressed_file_front_data), mode='rb') as f:
            file_front_data = f.read()

        with gzip.GzipFile(fileobj=io.BytesIO(compressed_file_left_data), mode='rb') as f:
            file_left_data = f.read()

        # Convert gender to numerical value
        gender_num = 1 if gender.lower() == 'male' else 0

        # Read images
        image_front = await process_image_async(file_front_data)
        image_left = await process_image_async(file_left_data)

        # with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_front, \
        #         tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_left:
        #     image_front.save(temp_front.name)
        #     image_left.save(temp_left.name)
        #
        #     results = predict(height, weight, temp_front.name, temp_left.name)
        #
        # os.remove(temp_front.name)
        # os.remove(temp_left.name)

        results = predict(height, weight, image_front, image_left, deeplab, device, model_wrist, model_waist, model_hip,
                          scaler, model_lock)

        # Calculate final metrics
        final_metrics = calculate_final_metrics(
            sex='male' if gender_num == 1 else 'female',
            neck_circumference=results['Neck'],
            waist_circumference=results['Waist'],
            hip_circumference=results['Hip'],
            height=height,
            weight=weight,
        )

        health_report = await generate_health_report(final_metrics, age, gender)

        response_content = {
            "final_metrics": final_metrics,
            "health_report": health_report
        }

        return JSONResponse(content=response_content)
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid Base64 encoding")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot identify image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
