import asyncio
import base64
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


@app.post("/convert-to-base64/")
async def convert_image_to_base64(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        file_data = await file.read()

        # Encode the file data to Base64
        base64_encoded = base64.b64encode(file_data).decode('utf-8')

        # Return the Base64 encoded string
        return JSONResponse(content={"base64_encoded": base64_encoded})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/")
async def predict_bfp_bmi_fmi(request: Request, predict_request: PredictRequest):
    try:

        # Extract the request details
        method = request.method
        url = str(request.url)
        headers = dict(request.headers)
        body = await request.json()

        # Generate the curl command
        curl_command = f"curl -X {method} '{url}'"

        # Add headers to the curl command
        for header, value in headers.items():
            if header.lower() not in ["content-length", "host"]:
                curl_command += f" -H '{header}: {value}'"

        # Add the body to the curl command if it exists
        if body:
            curl_command += f" -d '{json.dumps(body)}'"

        logger.info(f"Received request: {curl_command}")

        # Extract the relevant fields from the body
        height = predict_request.height
        weight = predict_request.weight
        age = predict_request.age
        gender = predict_request.gender

        # Decode Base64 strings
        file_front_data = base64.b64decode(predict_request.file_front)
        file_left_data = base64.b64decode(predict_request.file_left)

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
