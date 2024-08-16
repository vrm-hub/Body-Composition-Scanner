from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import gzip

from api.model_predictions import predict
from api.metrics_calculation import calculate_final_metrics
from api.open_ai import generate_health_report

app = FastAPI()


def compress_data(data):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(data.encode('utf-8'))
    return buf.getvalue()


@app.get("/")
async def health_check():
    return {"message": "The Health Check is successful"}


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

    health_report = await generate_health_report(final_metrics)

    response_content = {
        "final_metrics": final_metrics
    }

    return JSONResponse(content=response_content)
