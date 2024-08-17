import pandas as pd
import cv2
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
import joblib
from sklearn.preprocessing import StandardScaler

# Measurements file of subjects
df_measurements = pd.read_csv('train/measurements.csv')
df_measurements.head(5)

# Metadata for subjects
df_info = pd.read_csv('train/hwg_metadata.csv')
df_info.head(5)

# Subject to photo mapping
df_photo_map = pd.read_csv('train/subject_to_photo_map.csv')
df_photo_map.head(5)

# Merge Measurements with Metadata
df_subject_info_merged = pd.merge(df_measurements, df_info, on='subject_id', how='inner')
df_subject_info_merged.head(5)
df_photo_map = pd.merge(df_photo_map, df_subject_info_merged[['subject_id', 'height']], on='subject_id', how='inner')


# Method to return the image path of subjects of front view and left view
def return_image_path(photo_id):
    image_dir = 'train/mask'
    image_path_mask = os.path.join(image_dir, f'{photo_id}.png')

    image_dir = 'train/mask_left'
    image_path_maskleft = os.path.join(image_dir, f'{photo_id}.png')

    image = cv2.imread(image_path_maskleft)
    if image is None:
        print(f'Warning: Image {photo_id}.png not found')
    return [image_path_mask, image_path_maskleft]


# Extract features from imagges
def extract_silhouette_features(photo_id, subject_height):
    if '.png' in photo_id:
        image_ = [photo_id]
    else:
        image_ = return_image_path(photo_id=photo_id)
    features = {}
    loop = 0
    for image in image_:
        # Convert image to grayscale
        image = cv2.imread(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if loop == 0:
            loop = 1
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

            # Add features to dictionary
            scale = subject_height / h  # Use height of bounding box for scaling
            if loop == 2:
                features['area_left'] = area * (scale ** 2)
                features['perimeter_left'] = perimeter * scale
                features['aspect_ratio_left'] = aspect_ratio
                features['solidity_left'] = solidity
            else:
                features['area'] = area * (scale ** 2)
                features['perimeter'] = perimeter * scale
                features['aspect_ratio'] = aspect_ratio
                features['solidity'] = solidity

        loop = 2

    return features


# Extract image features and append
df_photo_map['metrics'] = df_photo_map.apply(lambda row: extract_silhouette_features(row['photo_id'], row['height']),
                                             axis=1)
df_photo_map.head(10)

# Extract each feature from the dict
df_metrics = pd.json_normalize(df_photo_map['metrics'])

# Map extracted features for each image wrt to subject
df_photo_map_extended = pd.concat([df_photo_map[['subject_id', 'photo_id']], df_metrics], axis=1)
df_photo_map_extended.head(5)

# Drop column photoid
df_photo_map_extended = df_photo_map_extended.drop(columns=['photo_id'])

# Avg the metrics for each subject since some subjects have multiple images in dataset
df_final_one = df_photo_map_extended.groupby('subject_id').mean().reset_index()
df_final_one.head(5)

# Merging info df with finalone to extract info of subjects
df_for_merge = df_subject_info_merged[['subject_id', 'weight_kg', 'height', 'wrist', 'waist', 'gender', 'hip']]

df = pd.merge(df_final_one, df_for_merge, on='subject_id', how='inner')

# Convert categorical col gender to numerical
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'male' else 0)

df.head(5)

# Normalize data
scaler = StandardScaler()

# Select the columns to normalize
columns_to_normalize = ['area', 'height', 'weight_kg', 'perimeter', 'aspect_ratio', 'solidity', 'area_left',
                        'perimeter_left', 'aspect_ratio_left', 'solidity_left']

# Fit and transform the data
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

scaler_file = "models/scaler.pkl"
joblib.dump(scaler, scaler_file)

df.head(5)


def train_and_save_model(df, target, model_name):
    model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    model2 = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model3 = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model4 = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

    ensemble_model = VotingRegressor([('rf', model1), ('xgb', model2), ('svr', model3), ('mlp', model4)])
    ensemble_model.fit(df.drop(columns=['subject_id', 'waist', 'wrist', 'hip']), df[[target]])

    model_file = f"models/{model_name}_ensemble_model.pkl"
    joblib.dump(ensemble_model, model_file)
    print(f"Model for {target} saved as {model_file}")

# Testing. Uncomment to train
# train_and_save_model(df, 'wrist', 'wrist')
# train_and_save_model(df, 'waist', 'waist')
# train_and_save_model(df, 'hip', 'hip')