from model_predictions import predict
from metrics_calculation import calculate_final_metrics


def main():
    height = 183  # in cm
    weight = 98  # in kg
    front_image_path = "front.jpg"
    left_image_path = "left.jpg"
    results = predict(height, weight, front_image_path, left_image_path)
    final_metrics = calculate_final_metrics('male', results['Neck'], results['Waist'], results['Hip'], height, weight)
    print(final_metrics)


if __name__ == "__main__":
    main()
