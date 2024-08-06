import torch
import cv2
import argparse
import json
from ultralytics import YOLO
import warnings

# Disable cudnn
torch.backends.cudnn.enabled = False

# Disable warnings
warnings.filterwarnings("ignore")

# Path to the YOLO model
MODEL_PATH = 'trained_model/90_best.pt'

def process_single_image(image_path):
    """
    Process a single image, performing detection, and displaying detection coordinates as JSON.

    Args:
    - image_path (str): Path to the image file.
    """
    # Load YOLO model
    model = YOLO(MODEL_PATH)  # Using the predefined path to the model

    # Load the image
    image = cv2.imread(image_path)

    # Perform detection on the image
    results = model(image)

    detected = False  # Flag indicating if any plates were detected

    # List to store detection coordinates
    detections = []

    # Process detection results
    for r in results:
        if hasattr(r, 'boxes'):
            boxes = r.boxes
            for b in boxes:
                xyxy = b.xyxy.cpu().numpy().astype(int).flatten()
                confidence = float(b.conf)
                class_id = int(b.cls)
                class_name = r.names[class_id]
                label = f"{class_name} {confidence:.2f}"

                if confidence > 0.35:
                    detected = True

                    # Store detection coordinates in a dictionary
                    conf = f'{confidence: .2f}'
                    detection_info = {
                        'class_name': class_name,
                        'confidence': conf,
                        'coordinates': {
                            'A': f"({xyxy[0]}, {xyxy[1]})",
                            'B': f"({xyxy[2]}, {xyxy[3]})"
                        }
                    }
                    detections.append(detection_info)

    # Display the detection coordinates as JSON
    print("Contents of JSON file:")
    print(json.dumps(detections, indent=4))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a single image with YOLO model and display detection coordinates as JSON.')
    parser.add_argument('image_path', type=str, help='Path to the image file')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    process_single_image(args.image_path)
