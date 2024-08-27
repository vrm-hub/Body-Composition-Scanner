import base64
import sys
import threading
import time

import torch
import logging

from api.utils import make_deeplab
import joblib

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io

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
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("controller")


async def process_image_async(image_data: bytes) -> Image.Image:
    # Open the image asynchronously
    return Image.open(io.BytesIO(image_data))


@app.get("/")
async def health_check():
    return {"message": "The Health Check is successful"}


@app.post("/create-request/")
async def create_request(height: float,
                         weight: float,
                         age: int,
                         gender: str,
                         file_front: UploadFile = File(...),
                         file_left: UploadFile = File(...)):
    front_encoded = await convert_image_to_base64(file_front)
    left_encoded = await convert_image_to_base64(file_left)

    body = {
        "height": height,
        "weight": weight,
        "age": age,
        "gender": gender,
        "file_front": front_encoded,
        "file_left": left_encoded
    }

    return body


@app.post("/convert-to-base64/")
async def convert_image_to_base64(file: UploadFile = File(...)) -> str:
    try:
        file_data = await file.read()
        return base64.b64encode(file_data).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/")
async def predict_bfp_bmi_fmi(predict_request: PredictRequest):
    try:
        start_time = time.time()

        height = predict_request.height
        weight = predict_request.weight
        age = predict_request.age
        gender = predict_request.gender

        file_front_data = base64.b64decode(predict_request.file_front)
        file_left_data = base64.b64decode(predict_request.file_left)

        gender_num = 1 if gender.lower() == 'male' else 0

        image_front = await process_image_async(file_front_data)
        image_left = await process_image_async(file_left_data)

        results = predict(height, weight, image_front, image_left, deeplab, device, model_wrist, model_waist, model_hip,
                          scaler, model_lock)

        final_metrics = calculate_final_metrics(
            sex='male' if gender_num == 1 else 'female',
            neck_circumference=results['Neck'],
            waist_circumference=results['Waist'],
            hip_circumference=results['Hip'],
            height=height,
            weight=weight,
        )

        health_report = await generate_health_report(final_metrics, age, gender)

        process_time = time.time() - start_time
        logger.info(f"Process time: {process_time:.4f} seconds")

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
