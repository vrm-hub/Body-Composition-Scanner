import pandas as pd
import threading


def predict_body_metrics(height, weight, front_image_path, left_image_path, deeplab, device, model_wrist, model_waist,
                         model_hip, scaler):
    from api.image_processing import preprocess, apply_deeplab, create_buff, access_stored_image
    from api.utils import extract_silhouette_features_for_inference

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


def predict(height, weight, image_front, image_left, deeplab, device, model_wrist, model_waist, model_hip, scaler,
            model_lock):
    # Use the global lock to synchronize access
    with model_lock:
        return predict_body_metrics(height, weight, image_front, image_left, deeplab, device, model_wrist,
                                    model_waist, model_hip, scaler)
