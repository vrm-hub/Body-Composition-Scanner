import base64
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
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
async def predict_bfp_bmi_fmi(request: PredictRequest):
    try:
        height = request.height
        weight = request.weight
        age = request.age
        gender = request.gender
        file_front_data = request.file_front
        file_left_data = request.file_left

        # Decode Base64 strings
        file_front_data = base64.b64decode(request.file_front)
        file_left_data = base64.b64decode(request.file_left)

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
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid Base64 encoding")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot identify image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
