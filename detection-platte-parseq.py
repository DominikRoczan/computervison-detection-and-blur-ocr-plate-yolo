from ultralytics import YOLO
from torchvision import transforms
import torch
from strhub.data.module import SceneTextDataModule

import cv2
import json
import argparse
import time
from PIL import Image


# Disable cudnn for stability (optional)
torch.backends.cudnn.enabled = True

# Ścieżka do modelu YOLO
MODEL_PATH = '../trained_model/90_best.pt'

# Zmienna globalna do przechowywania modeli
yolo_model = None
parseq_model = None


# Sprawdzanie dostępności GPU
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_yolo_model():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO(MODEL_PATH).to(device)
        print(f"YOLO model loaded on {device}")
    return yolo_model


def load_ocr_model():
    global parseq_model
    if parseq_model is None:
        parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
        print(f"Parseq model loaded on {device}")
    return parseq_model

def perform_ocr(img, parseq, img_transform):
    """Perform OCR on a given image and return the decoded text."""
    try:
        start_time = time.time()

        # Preprocess the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = img_transform(img_pil).unsqueeze(0).to(device)

        preprocess_ocr_time = time.time() - start_time

        # Perform prediction
        inference_start_time = time.time()
        with torch.no_grad():
            logits = parseq(img_tensor)
        pred = logits.softmax(-1)
        label, _ = parseq.tokenizer.decode(pred)

        inference_ocr_time = time.time() - inference_start_time
        total_ocr_time = preprocess_ocr_time + inference_ocr_time

        return label[0], preprocess_ocr_time, inference_ocr_time, total_ocr_time
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None, 0, 0, 0


def process_single_image(image_path):
    """
    Process a single image, performing detection, and displaying detection coordinates as JSON.
    """
    total_start_time = time.time()  # Start of full process
    print("Starting full image processing...")

    # YOLO model loading
    yolo_model_start = time.time()
    model = load_yolo_model()  # Załadowanie modelu YOLO
    yolo_model_load_time = time.time() - yolo_model_start
    print(f"YOLO model loading time: {yolo_model_load_time:.4f} s")

    # Load OCR model
    parseq_start = time.time()
    parseq = load_ocr_model()  # Załadowanie modelu OCR (PARSeq)
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    parseq_load_time = time.time() - parseq_start
    print(f"PARSeq OCR model loading time: {parseq_load_time:.4f} s")

    # Image loading
    image_load_start = time.time()
    image = cv2.imread(image_path)
    image_load_time = time.time() - image_load_start
    print(f"Image loading time: {image_load_time:.4f} s")

    # YOLO inference
    yolo_inference_start = time.time()
    results = model(image)
    yolo_inference_time = time.time() - yolo_inference_start
    print(f"YOLO inference time: {yolo_inference_time:.4f} s")

    # Process detection results
    detection_start = time.time()
    detections = []
    for r in results:
        if hasattr(r, 'boxes'):
            boxes = r.boxes
            for b in boxes:
                xyxy = b.xyxy.cpu().numpy().astype(int).flatten()
                confidence = float(b.conf)
                class_id = int(b.cls)
                class_name = r.names[class_id]

                if confidence > 0.35:
                    x1, y1, x2, y2 = xyxy
                    cropped_img = image[y1:y2, x1:x2]

                    detection_info = {
                        "className": class_name,
                        "confidence": f"{confidence:.2f}",
                        "coordinates": {
                            "A": f"({x1}, {y1})",
                            "B": f"({x2}, {y2})"
                        }
                    }

                    # Perform OCR if class is not 'Face'
                    if class_name != "Twarz":
                        ocr_result, preprocess_ocr_time, inference_ocr_time, total_ocr_time = perform_ocr(
                            cropped_img, parseq, img_transform)
                        if ocr_result:
                            detection_info["Numer tablicy"] = ocr_result
                            detection_info["OCR_times"] = {
                                "preprocess_ocr_time": f"{preprocess_ocr_time:.4f} s",
                                "inference_ocr_time": f"{inference_ocr_time:.4f} s",
                                "total_ocr_time": f"{total_ocr_time:.4f} s"
                            }
                    detections.append(detection_info)

    detection_processing_time = time.time() - detection_start
    print(f"Detection and OCR processing time: {detection_processing_time:.4f} s")

    # Total processing time
    total_processing_time = time.time() - total_start_time

    # Prepare output data
    output_data = {
        "detections": detections,
        "timing": {
            "YOLO_model_loading_time": f"{yolo_model_load_time:.4f} s",
            "PARSeq_model_loading_time": f"{parseq_load_time:.4f} s",
            "image_loading_time": f"{image_load_time:.4f} s",
            "YOLO_inference_time": f"{yolo_inference_time:.4f} s",
            "detection_processing_time": f"{detection_processing_time:.4f} s",
            "total_processing_time": f"{total_processing_time:.4f} s"
        }
    }

    # Output JSON
    print(json.dumps(output_data, indent=4))


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process a single image with YOLO model and perform OCR on detected regions.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    return parser.parse_args()


def main():
    """Main function to initialize models and process the input image."""
    args = parse_arguments()

    # Process the input image
    process_single_image(args.image_path)


if __name__ == "__main__":
    main()
