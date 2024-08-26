import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import gzip

from api.model_predictions import predict
from api.metrics_calculation import calculate_final_metrics
from api.open_ai import generate_health_report
from api.request_model import PredictRequest

app = FastAPI()


def compress_data(data):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(data.encode('utf-8'))
    return buf.getvalue()


@app.get("/")
async def health_check():
    return {"message": "The Health Check is successful"}


async def predict_bfp_bmi_fmi(request: PredictRequest):
    height = request.height
    weight = request.weight
    age = request.age
    gender = request.gender
    file_front_data = request.file_front
    file_left_data = request.file_left

    # Convert gender to numerical value
    gender_num = 1 if gender.lower() == 'male' else 0

    # Read images
    image_front = Image.open(io.BytesIO(file_front_data))
    image_left = Image.open(io.BytesIO(file_left_data))

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_front, \
            tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_left:
        image_front.save(temp_front.name)
        image_left.save(temp_left.name)

        results = predict(height, weight, temp_front.name, temp_left.name)

    os.remove(temp_front.name)
    os.remove(temp_left.name)

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
