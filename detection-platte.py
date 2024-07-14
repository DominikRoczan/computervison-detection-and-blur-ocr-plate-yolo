import os
import cv2
import argparse
import json
from ultralytics import YOLO

# Stałe ustawienia
MODEL_PATH = 'trained_model/90_best.pt'  # Ścieżka do modelu YOLO
OUTPUT_DIR = 'output_blurred'  # Katalog wynikowy dla zdjęć po przetworzeniu

def process_single_image(image_path):
    """
    Process a single image, performing detection, saving the result, and storing detection coordinates in a JSON file.

    Args:
    - image_path (str): Path to the image file.
    """
    # Load YOLO model
    model = YOLO(MODEL_PATH)  # Using the predefined path to the model

    # Create output directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

                    # Draw red bounding box and put confidence score
                    cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)

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

                    # Print coordinates of the bounding box
                    #print(f"Class: {class_name}, Confidence: {confidence:.2f}")
                    #print(f"Bounding Box Coordinates: A-({xyxy[0]}, {xyxy[1]}), B-({xyxy[2]}, {xyxy[3]})")

    # Save detection coordinates to JSON file
    json_output_path = os.path.join(OUTPUT_DIR, f'{os.path.splitext(os.path.basename(image_path))[0]}.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(detections, json_file, indent=4)

    # Save modified image to the appropriate folder
    if detected:
        final_image_path = os.path.join(OUTPUT_DIR, f'{os.path.splitext(os.path.basename(image_path))[0]}_Blur.jpg')
        cv2.imwrite(final_image_path, image)
    else:
        final_image_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
        cv2.imwrite(final_image_path, image)
        print(f"No license plates were detected in {image_path}.")

    # Display the processed image
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Open and print the contents of the JSON file
    if os.path.exists(json_output_path):
        with open(json_output_path, 'r') as json_file:
            json_data = json.load(json_file)
            print("Contents of JSON file:")
            print(json.dumps(json_data, indent=4))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a single image with YOLO model, display the result, and save detection coordinates in JSON.')
    parser.add_argument('image_path', type=str, help='Path to the image file')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    process_single_image(args.image_path)
