from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import pandas as pd

# Import functions from body-scan.py
from body_scan import predict, calculate_final_metrics

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}

@app.post("/predict/")
async def predict_bfp_bmi_fmi(
    file_front: UploadFile = File(...),
    file_left: UploadFile = File(...),
    height: int = 183,
    weight: int = 98,
    gender: str = 'male'
):
    # Convert gender to numerical value
    gender_num = 1 if gender.lower() == 'male' else 0

    # Read images
    image_front = Image.open(io.BytesIO(await file_front.read()))
    image_left = Image.open(io.BytesIO(await file_left.read()))

    # Save images temporarily
    front_image_path = "temp_front.jpg"
    left_image_path = "temp_left.jpg"
    image_front.save(front_image_path)
    image_left.save(left_image_path)

    # Use the functions from body-scan.py to perform calculations
    results = predict(height, weight, front_image_path, left_image_path)

    # Calculate final metrics
    final_metrics = calculate_final_metrics(
        sex='male' if gender_num == 1 else 'female',
        neck_circumference=results['Neck'],
        waist_circumference=results['Waist'],
        hip_circumference=results['Hip'],
        height=height,
        weight=weight
    )

    return JSONResponse(content=final_metrics)
