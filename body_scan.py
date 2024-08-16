import io
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
import joblib


# Method to extract features for inference images
def extract_silhouette_features_for_inference(image, subject_height):
    image_array = np.array(image)
    features ={}
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to get binary silhouette
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # Iterate through contours
    for contour in contours:
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0.0
        
        # Calculate solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0.0

        scale = subject_height / h  # Use height of bounding box for scaling
        features['area'] = area * (scale ** 2)
        features['perimeter'] = perimeter * scale
        features['aspect_ratio'] = aspect_ratio
        features['solidity'] = solidity

    return features
            


# load prebuilt Resnet model 
def make_deeplab(device):
    deeplab = deeplabv3_resnet101(pretrained=True).to(device)
    deeplab.eval()
    return deeplab

# Generate masked images
deeplab_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_body_metrics(height, weight, front_image_path, left_image_path, deeplab, device, model_wrist, model_waist, model_hip, scaler):
    
    # resize images
    def preprocess(f_name):
        img_orig = cv2.imread(f_name, 1)
        k = min(1.0, 1024/max(img_orig.shape[0], img_orig.shape[1]))
        img = cv2.resize(img_orig, None, fx=k, fy=k, interpolation=cv2.INTER_LANCZOS4)
        return img
    
    def apply_deeplab(deeplab, img, device):
        input_tensor = deeplab_preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = deeplab(input_batch.to(device))['out'][0]
        output_predictions = output.argmax(0).cpu().numpy()
        return (output_predictions == 15)


    # Buff to store image temp
    def create_buff(mask):
        buf = io.BytesIO()
        plt.imsave(buf, mask, cmap="gray", format="png")
        buf.seek(0)
        return buf

    # Access the image stored in buff
    def access_stored_image(buffer):
        # Open the image from the BytesIO buffer
        image = Image.open(buffer)
        return image

    # Process front and left images
    mask_front = apply_deeplab(deeplab, preprocess(front_image_path), device)
    stored_image_front = access_stored_image(create_buff(mask_front))
    dict1 = extract_silhouette_features_for_inference(stored_image_front, height)

    mask_left = apply_deeplab(deeplab, preprocess(left_image_path), device)
    stored_image_left = access_stored_image(create_buff(mask_left))
    dict2 = extract_silhouette_features_for_inference(stored_image_left, height)

    # Combine features into a single dictionary
    combined_dict = {
        'area': dict1['area'],
        'perimeter': dict1['perimeter'],
        'aspect_ratio': dict1['aspect_ratio'],
        'solidity': dict1['solidity'],
        'area_left': dict2['area'],
        'perimeter_left': dict2['perimeter'],
        'aspect_ratio_left': dict2['aspect_ratio'],
        'solidity_left': dict2['solidity']
    }

    # Convert combined features to DataFrame
    df_inference = pd.DataFrame([combined_dict])

    # Merge with image metrics
    df_inference['weight_kg'] = weight
    df_inference['height'] = height
    df_inference['gender'] = 1  # Assuming gender is constant

    # Normalize the features with respect to the test data
    mean = scaler.mean_
    scale = scaler.scale_

    for i, col in enumerate(['area', 'height', 'weight_kg', 'perimeter', 'aspect_ratio', 'solidity',
                             'area_left', 'perimeter_left', 'aspect_ratio_left', 'solidity_left']):
        df_inference[col] = (df_inference[col] - mean[i]) / scale[i]

    # Predict wrist, waist, and hip circumferences
    wrist_circumference = model_wrist.predict(df_inference)[0]
    waist_circumference = model_waist.predict(df_inference)[0]
    hip_circumference = model_hip.predict(df_inference)[0]

    # Calculate neck circumference based on wrist circumference
    neck_circumference = 2.64 + 1.94 * wrist_circumference

    return {
        'Neck': neck_circumference,
        'Waist': waist_circumference,
        'Hip': hip_circumference
    }

def calculate_final_metrics(sex, neck_circumference, waist_circumference, hip_circumference, height, weight):

    def calculate_bfp(sex, neck_circumference, waist_circumference, hip_circumference, height):
        """
        Calculate Body Fat Percentage using the Navy Body Fat formula.
        """
        if sex.lower() == 'male':
            bfp = 86.010 * math.log10(waist_circumference - neck_circumference) - 70.041 * math.log10(height) + 36.76
        elif sex.lower() == 'female':
            bfp = 163.205 * math.log10(waist_circumference + hip_circumference - neck_circumference) - 97.684 * math.log10(height) - 78.387
        else:
            raise ValueError("Sex must be 'male' or 'female'")
        return bfp

    def classify_fat_types(weight, bfp):
        """
        Classify fat types into essential, beneficial, and unbeneficial fats.
        """
        fm = weight * (bfp / 100)
        if sex.lower() == 'male':
            essential_fat = max(0.02 * weight, min(0.05 * weight, fm))
        elif sex.lower() == 'female':
            essential_fat = max(0.10 * weight, min(0.13 * weight, fm))
        beneficial_fat = 0.15 * weight  # 15% of total body weight for beneficial fat
        unbeneficial_fat = fm - (essential_fat + beneficial_fat)
        return essential_fat, beneficial_fat, unbeneficial_fat

    def calculate_lean_mass(weight, fm):
        """
        Calculate Lean Mass.
        """
        return weight - fm

    def calculate_indices(lean_mass, fm, height):
        """
        Calculate Lean Mass Index (LMI) and Fat Mass Index (FMI).
        """
        height_m = height / 100  # convert height to meters
        lmi = lean_mass / (height_m ** 2)
        fmi = fm / (height_m ** 2)
        return lmi, fmi

    def calculate_rmr(lean_mass):
        """
        Calculate Resting Metabolic Rate (RMR) using the Katch-McArdle formula.
        """
        return 370 + (21.6 * lean_mass)

    # Calculations
    bfp = calculate_bfp(sex, neck_circumference, waist_circumference, hip_circumference, height)
    fm = weight * (bfp / 100)
    essential_fat, beneficial_fat, unbeneficial_fat = classify_fat_types(weight, bfp)
    lean_mass = calculate_lean_mass(weight, fm)
    lmi, fmi = calculate_indices(lean_mass, fm, height)
    rmr = calculate_rmr(lean_mass)

    metrics = {
        'Neck': neck_circumference,
        'Waist': waist_circumference,
        'Hip': hip_circumference,
        'Height': height,
        'Weight': weight,
        'BFP': bfp,
        'Essential_Fat': essential_fat,
        'Beneficial_Fat': beneficial_fat,
        'Unbeneficial_Fat': unbeneficial_fat,
        'Lean_Mass': lean_mass,
        'LMI': lmi,
        'FMI': fmi,
        'RMR': rmr
    }

    # Print Results
    print(f"Body Fat Percentage (BFP): {bfp:.2f}%")
    print(f"Essential Fat: {essential_fat:.2f} kg")
    print(f"Beneficial Fat: {beneficial_fat:.2f} kg")
    print(f"Unbeneficial Fat: {unbeneficial_fat:.2f} kg")
    print(f"Lean Mass: {lean_mass:.2f} kg")
    print(f"Lean Mass Index (LMI): {lmi:.2f} kg/m^2")
    print(f"Fat Mass Index (FMI): {fmi:.2f} kg/m^2")
    print(f"Resting Metabolic Rate (RMR): {rmr:.2f} kcal/day")

    return metrics

def predict(height, weight, front_image_path, left_image_path):
    device = torch.device("cpu")
    deeplab = make_deeplab(device)
    
    model_wrist = joblib.load("models/wrist_ensemble_model.pkl")
    model_waist = joblib.load("models/waist_ensemble_model.pkl")
    model_hip = joblib.load("models/hip_ensemble_model.pkl")

    scaler_file = "models/scaler.pkl"           
    scaler = joblib.load(scaler_file)

    return predict_body_metrics(height, weight, front_image_path, left_image_path, deeplab, device, model_wrist, model_waist, model_hip, scaler)
    

# Testing
# height = 183  # in cm
# weight = 98   # in kg
# front_image_path = "front.jpg"
# left_image_path = "left.jpg"
# results = predict(height, weight, front_image_path, left_image_path)
# calculate_final_metrics('male', results['Neck'], results['Waist'], results['Hip'], height, weight)


