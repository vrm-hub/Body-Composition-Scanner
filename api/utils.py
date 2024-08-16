import cv2
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101


# Method to extract features for inference images
def extract_silhouette_features_for_inference(image, subject_height):
    image_array = np.array(image)
    features = {}
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