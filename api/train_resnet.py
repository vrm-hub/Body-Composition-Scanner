import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, Callback
import pandas as pd
import os
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import time

# Enable logging of device placement (Optional)
tf.debugging.set_log_device_placement(True)

# Load datasets
df_measurements = pd.read_csv('../train/measurements.csv')
df_info = pd.read_csv('../train/hwg_metadata.csv')
df_photo_map = pd.read_csv('../train/subject_to_photo_map.csv')

# Merge datasets
df_subject_info_merged = pd.merge(df_measurements, df_info, on='subject_id', how='inner')
df_photo_map = pd.merge(df_photo_map, df_subject_info_merged[['subject_id', 'height']], on='subject_id', how='inner')


# Combine image path retrieval
def return_image_paths(photo_id):
    image_mask = os.path.join('../train/mask', f'{photo_id}.png')
    image_mask_left = os.path.join('../train/mask_left', f'{photo_id}.png')
    return [image_mask, image_mask_left]


# Load and preprocess images
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size, color_mode='rgb')
    image = img_to_array(image)
    image /= 255.0  # Normalize to [0, 1]
    return image


# Extract image features and append
def prepare_image_data(df):
    images = []
    for _, row in df.iterrows():
        paths = return_image_paths(row['photo_id'])
        img1 = load_and_preprocess_image(paths[0])
        img2 = load_and_preprocess_image(paths[1])

        # Combine and reduce to 3 channels
        combined_img = np.mean(np.stack([img1, img2], axis=-1), axis=-1)  # Average to reduce to 3 channels

        images.append(combined_img)
    return np.array(images)


# Prepare images and measurements
X_images = prepare_image_data(df_photo_map)
measurements = df_photo_map[['subject_id']].merge(df_subject_info_merged, on='subject_id')

# Convert gender to numeric values
measurements['gender'] = measurements['gender'].map({'male': 1, 'female': 0})

# Extract features
X_measurements = measurements[['weight_kg', 'height', 'gender']].values

# Normalize measurements
scaler = StandardScaler()
X_measurements = scaler.fit_transform(X_measurements)
joblib.dump(scaler, "../models-test/scaler.pkl")

# Targets
y_wrist = measurements['wrist'].values
y_waist = measurements['waist'].values
y_hip = measurements['hip'].values

# Train/test split
X_train_img, X_test_img, X_train_meas, X_test_meas, y_train_wrist, y_test_wrist, y_train_waist, y_test_waist, y_train_hip, y_test_hip = train_test_split(
    X_images, X_measurements, y_wrist, y_waist, y_hip, test_size=0.2, random_state=42
)


# Build the model using ResNet50 with a custom input layer
def build_resnet50_model(input_shape_img, input_shape_meas):
    # Custom input for the image data
    img_input = Input(shape=input_shape_img)

    # Custom first layer to accommodate 6 channels
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', name='custom_conv1')(img_input)

    # Load the ResNet50 model without the top layers and without pre-trained weights
    base_model = ResNet50(weights=None, include_top=False, input_tensor=x)  # Set weights=None

    # Continue with the rest of the ResNet50 layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    # Measurements input
    meas_input = Input(shape=input_shape_meas)

    # Combine features
    combined = Concatenate()([x, meas_input])
    combined = Dense(256, activation='relu')(combined)
    combined = Dense(128, activation='relu')(combined)

    # Output layers
    wrist_output = Dense(1, name='wrist')(combined)
    waist_output = Dense(1, name='waist')(combined)
    hip_output = Dense(1, name='hip')(combined)

    model = Model(inputs=[img_input, meas_input], outputs=[wrist_output, waist_output, hip_output])

    return model


# Model instantiation
model = build_resnet50_model(X_train_img.shape[1:], X_train_meas.shape[1:])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# TensorBoard log directory
log_dir = "../logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# Custom callback to measure time per epoch
class TimingCallback(Callback):
    def __init__(self):
        super(TimingCallback, self).__init__()
        self.logs = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        self.logs.append(elapsed_time)
        print(f"Epoch {epoch + 1}: time taken - {elapsed_time:.2f} seconds")


timing_callback = TimingCallback()

# Train the model with callbacks
history = model.fit(
    [X_train_img, X_train_meas],
    [y_train_wrist, y_train_waist, y_train_hip],
    validation_data=([X_test_img, X_test_meas], [y_test_wrist, y_test_waist, y_test_hip]),
    epochs=20,
    batch_size=32,
    callbacks=[tensorboard_callback, timing_callback]
)

# Save the model
model.save("../models-test/resnet50_model.h5")
print("ResNet-50 model saved.")

# Predict on the test set
y_pred_wrist, y_pred_waist, y_pred_hip = model.predict([X_test_img, X_test_meas])

# Calculate and display metrics
mae_wrist = mean_absolute_error(y_test_wrist, y_pred_wrist)
r2_wrist = r2_score(y_test_wrist, y_pred_wrist)

mae_waist = mean_absolute_error(y_test_waist, y_pred_waist)
r2_waist = r2_score(y_test_waist, y_pred_waist)

mae_hip = mean_absolute_error(y_test_hip, y_pred_hip)
r2_hip = r2_score(y_test_hip, y_pred_hip)

print(f"Wrist - MAE: {mae_wrist:.4f}, R²: {r2_wrist:.4f}")
print(f"Waist - MAE: {mae_waist:.4f}, R²: {r2_waist:.4f}")
print(f"Hip - MAE: {mae_hip:.4f}, R²: {r2_hip:.4f}")
