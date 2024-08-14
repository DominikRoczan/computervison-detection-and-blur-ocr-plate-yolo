import torch
import cv2
import argparse
import json
import os
from ultralytics import YOLO
from strhub.data.module import SceneTextDataModule
import warnings
import pytorch_lightning as pl
from torchvision import transforms

# Disable cudnn
torch.backends.cudnn.enabled = False

# Disable warnings
warnings.filterwarnings("ignore")

# Path to the YOLO model
MODEL_PATH = './trained_model/90_best.pt'


def load_ocr_model():
    """Load the pre-trained PARSeq model and image transformations for OCR."""
    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    return parseq, img_transform


def perform_ocr(img, parseq, img_transform):
    """Perform OCR on a given image and return the decoded text."""
    try:
        # Preprocess the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Transform to PIL Image and then to Tensor
        img_pil = transforms.ToPILImage()(img)
        img_tensor = img_transform(img_pil).unsqueeze(0)

        # print(f"Image shape after transformation: {img_tensor.shape}")
        # print(f"Image type after transformation: {type(img_tensor)}")

        # Perform prediction
        with torch.no_grad():
            logits = parseq(img_tensor)
        pred = logits.softmax(-1)
        label, _ = parseq.tokenizer.decode(pred)

        return label[0]
    except Exception as e:
        print(f'Error during OCR: {e}')
        return None


def process_single_image(image_path, parseq, img_transform):
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

    detected = False  # Flag indicating if any objects were detected

    # List to store detection coordinates and OCR results
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

                    # Crop the detected region for OCR
                    x1, y1, x2, y2 = xyxy
                    cropped_img = image[y1:y2, x1:x2]

                    # Perform OCR on the cropped image
                    ocr_result = perform_ocr(cropped_img, parseq, img_transform)

                    # Store detection and OCR information in a dictionary
                    conf = f'{confidence: .2f}'
                    detection_info = {
                        'class_name': class_name,
                        'confidence': conf,
                        'coordinates': {
                            'A': f"({xyxy[0]}, {xyxy[1]})",
                            'B': f"({xyxy[2]}, {xyxy[3]})"
                        },
                        'ocr_plate': ocr_result  # Add OCR result
                    }
                    detections.append(detection_info)

    # Print the detection and OCR results as JSON
    print(json.dumps(detections, indent=4))


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Process a single image with YOLO model and perform OCR on detected regions.')
    parser.add_argument('image_path', type=str, help='Path to the image file')

    return parser.parse_args()


def main():
    """Main function to load models, process the image, and display results."""
    args = parse_arguments()

    # Load OCR model and transformations
    parseq, img_transform = load_ocr_model()

    # Process the image with detection and OCR
    process_single_image(args.image_path, parseq, img_transform)


if __name__ == "__main__":
    main()
