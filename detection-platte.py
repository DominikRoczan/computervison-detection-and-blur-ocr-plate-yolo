from ultralytics import YOLO
import torch
from torchvision import transforms
from strhub.data.module import SceneTextDataModule
import cv2
import sys
import argparse
import os
import logging
import json

logging.getLogger('ultralytics').setLevel(logging.ERROR)
sys.stderr = open(os.devnull, 'w')

device = "cpu"

MODEL_PATH = './trained_model/train23/weights/Plates_Faces.pt'

yolo_model = None
parseq_model = None
img_transform = None

def load_yolo_model():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO(MODEL_PATH).to(device)
    return yolo_model

def load_ocr_model():
    global parseq_model, img_transform
    if parseq_model is None:
        parseq_model = torch.hub.load('baudm/parseq', 'parseq_tiny', pretrained=True).eval().to(device)
        img_transform = SceneTextDataModule.get_transform(parseq_model.hparams.img_size)
    return parseq_model, img_transform

def perform_ocr(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = transforms.ToPILImage()(img)
        img_tensor = img_transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = parseq_model(img_tensor)
        pred = logits.softmax(-1)
        label, _ = parseq_model.tokenizer.decode(pred)

        return label[0]
    except:
        return None

def process_single_image(image_path, yolo_model, parseq_model):
    image = cv2.imread(image_path)
    results = yolo_model(image)
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

    output_data = {"detections": detections}
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

if __name__ == "__main__":
    main()
