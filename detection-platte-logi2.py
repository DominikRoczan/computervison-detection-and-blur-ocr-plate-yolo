import time

# Measure time from the start of the script
script_start_time = time.perf_counter()

import argparse
from ultralytics import YOLO
from torchvision import transforms
from strhub.data.module import SceneTextDataModule

import torch
import cv2

import sys
import os
import warnings
import logging
import json

print("⏳ Script has started...")
print(f"⏱️ Total execution time (from script start to imports): {time.perf_counter() - script_start_time:.4f} s")

# Disable warnings, logs, and redirect stdout and stderr to null
warnings.filterwarnings("ignore")
logging.getLogger('ultralytics').setLevel(logging.ERROR)
sys.stderr = open(os.devnull, 'w')

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# Path to the YOLO model
MODEL_PATH = './trained_model/Plates_Faces.pt'
# MODEL_PATH = '../trained_model/train23/weights/Plates_Faces.pt'

yolo_model = None
parseq_model = None
img_transform = None


def load_yolo_model():
    """Load the YOLO model"""
    global yolo_model
    if yolo_model is None:
        yolo_model_start_time = time.perf_counter()
        yolo_model = YOLO(MODEL_PATH).to(device)
        yolo_model_end_time = time.perf_counter()
        print(f"✅ YOLO model loaded on {device}")
        print(f"⏱️ YOLO model loading time: {yolo_model_end_time - yolo_model_start_time:.4f} s")
    return yolo_model


def load_ocr_model():
    """Load the OCR model (PARSeq)"""
    global parseq_model, img_transform
    if parseq_model is None:
        ocr_model_start_time = time.perf_counter()
        parseq_model = torch.hub.load('baudm/parseq', 'parseq_tiny', pretrained=True).eval().to(device)
        img_transform = SceneTextDataModule.get_transform(parseq_model.hparams.img_size)
        ocr_model_end_time = time.perf_counter()
        print(f"✅ OCR model (Parseq) loaded on {device}")
        print(f"⏱️ OCR model loading time: {ocr_model_end_time - ocr_model_start_time:.4f} s")
    return parseq_model, img_transform


def perform_ocr(img):
    """Perform OCR on the given image"""
    try:
        ocr_start_time = time.perf_counter()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = transforms.ToPILImage()(img)
        img_tensor = img_transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = parseq_model(img_tensor)
        pred = logits.softmax(-1)
        label, _ = parseq_model.tokenizer.decode(pred)

        ocr_time = time.perf_counter() - ocr_start_time
        print(f"🔡 OCR processing time: {ocr_time:.4f} s")
        return label[0]
    except Exception as e:
        print(f"❌ Error during OCR: {e}")
        return None


def process_single_image(image_path, yolo_model, parseq_model):
    """Process a single image"""
    total_start_time = time.perf_counter()
    print("🚀 Starting full image processing...")

    image_load_start = time.perf_counter()
    image = cv2.imread(image_path)
    image_load_time = time.perf_counter() - image_load_start
    print(f"🖼️ Image loading time: {image_load_time:.4f} s")

    yolo_inference_start = time.perf_counter()
    results = yolo_model(image)
    yolo_inference_time = time.perf_counter() - yolo_inference_start
    print(f"🔍 YOLO inference time: {yolo_inference_time:.4f} s")

    detection_start = time.perf_counter()
    detections = []

    for r in results:
        if hasattr(r, 'boxes'):
            for b in r.boxes:
                xyxy = b.xyxy.cpu().numpy().astype(int).flatten()
                confidence = float(b.conf)
                class_id = int(b.cls)
                class_name = r.names[class_id]

                if confidence > 0.35:
                    x1, y1, x2, y2 = xyxy
                    detection_info = {
                        "className": class_name,
                        "confidence": f"{confidence:.2f}",
                        "coordinates": {
                            "A": f"({x1}, {y1})",
                            "B": f"({x2}, {y2})"
                        }
                    }

                    if class_name != "Twarz":
                        cropped_img = image[y1:y2, x1:x2]
                        ocr_result = perform_ocr(cropped_img)
                        if ocr_result:
                            detection_info["Plate_number"] = ocr_result

                    detections.append(detection_info)

    detection_processing_time = time.perf_counter() - detection_start
    print(f"📊 Detection processing time: {detection_processing_time:.4f} s")

    total_processing_time = time.perf_counter() - total_start_time

    output_data = {
        "detections": detections,
        "timing": {
            "image_loading_time": f"{image_load_time:.4f} s",
            "YOLO_inference_time": f"{yolo_inference_time:.4f} s",
            "detection_processing_time": f"{detection_processing_time:.4f} s",
            "total_processing_time": f"{total_processing_time:.4f} s"
        }
    }
    print(json.dumps(output_data, indent=4))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Object detection with OCR.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    return parser.parse_args()


def main():
    args = parse_arguments()
    global yolo_model, parseq_model
    yolo_model = load_yolo_model()
    parseq_model, _ = load_ocr_model()
    process_single_image(args.image_path, yolo_model, parseq_model)
    script_end_time = time.perf_counter()
    print(f"⏱️ Total execution time (from script start to end): {script_end_time - script_start_time:.4f} s")


if __name__ == "__main__":
    main()
