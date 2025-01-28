from ultralytics import YOLO
from torchvision import transforms
import torch
import torch.nn.functional as F
from strhub.data.module import SceneTextDataModule

import cv2
import json
import argparse
import time
from PIL import Image


# 🔹 Ustawienia CUDA dla lepszej wydajności
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# 🔹 Ścieżka do modelu YOLO
MODEL_PATH = './trained_model/Plates_Faces.pt'
# MODEL_PATH = '../trained_model/train23/weights/Plates_Faces.pt'


# 🔹 Zmienna globalna do przechowywania modeli
yolo_model = None
parseq_model = None

# 🔹 Sprawdzanie dostępności GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


def load_yolo_model():
    """Ładowanie modelu YOLO"""
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO(MODEL_PATH).to(device)
        print(f"✅ YOLO model loaded on {device}")
    return yolo_model


def load_ocr_model():
    """Ładowanie modelu OCR (PARSeq)"""
    global parseq_model
    if parseq_model is None:
        parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)
        print(f"✅ PARSeq model loaded on {device}")
    return parseq_model


def perform_ocr_batch(images, parseq, img_transform):
    """OCR dla batcha obrazów"""
    try:
        start_time = time.time()

        # 🔹 Konwersja do tensora
        tensors = [img_transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device) for img in images]
        batch_images = torch.cat(tensors, dim=0)

        preprocess_ocr_time = time.time() - start_time

        # 🔹 OCR batch processing
        inference_start_time = time.time()
        with torch.no_grad():
            logits = parseq(batch_images)
        pred = F.softmax(logits, dim=-1)
        labels, _ = parseq.tokenizer.decode(pred)

        inference_ocr_time = time.time() - inference_start_time
        total_ocr_time = preprocess_ocr_time + inference_ocr_time

        return labels, preprocess_ocr_time, inference_ocr_time, total_ocr_time
    except Exception as e:
        print(f"❌ Error during OCR: {e}")
        return None, 0, 0, 0


def process_single_image(image_path, yolo_model, parseq):
    """Przetwarzanie pojedynczego obrazu"""
    total_start_time = time.time()
    print("🚀 Starting full image processing...")

    # 🔹 Ładowanie obrazu
    image_load_start = time.time()
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1280, 720))  # Resize dla szybszego YOLO
    image_load_time = time.time() - image_load_start
    print(f"🖼️ Image loading time: {image_load_time:.4f} s")

    # 🔹 YOLO wykrywanie
    yolo_inference_start = time.time()
    results = yolo_model(image)
    yolo_inference_time = time.time() - yolo_inference_start
    print(f"🔍 YOLO inference time: {yolo_inference_time:.4f} s")

    # 🔹 Procesowanie detekcji
    detection_start = time.time()
    detections = []
    ocr_images = []
    ocr_positions = []
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

    for r in results:
        if hasattr(r, 'boxes'):
            for b in r.boxes:
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

                    # 🔹 OCR tylko dla tablic
                    if class_name != "Twarz":
                        ocr_images.append(cropped_img)
                        ocr_positions.append(detection_info)

                    detections.append(detection_info)

    detection_processing_time = time.time() - detection_start
    print(f"📊 Detection processing time: {detection_processing_time:.4f} s")

    # 🔹 OCR (przetwarzanie batcha)
    if ocr_images:
        ocr_results, preprocess_ocr_time, inference_ocr_time, total_ocr_time = perform_ocr_batch(
            ocr_images, parseq, img_transform
        )

        for i, ocr_text in enumerate(ocr_results):
            ocr_positions[i]["Numer tablicy"] = ocr_text
            ocr_positions[i]["OCR_times"] = {
                "preprocess_ocr_time": f"{preprocess_ocr_time:.4f} s",
                "inference_ocr_time": f"{inference_ocr_time:.4f} s",
                "total_ocr_time": f"{total_ocr_time:.4f} s"
            }

    # 🔹 Czas całkowity
    total_processing_time = time.time() - total_start_time

    # 🔹 Wynik JSON
    output_data = {
        "detections": detections,
        "timing": {
            "image_loading_time": f"{image_load_time:.4f} s",
            "YOLO_inference_time": f"{yolo_inference_time:.4f} s",
            "detection_processing_time": f"{detection_processing_time:.4f} s",
            "total_processing_time": f"{total_processing_time:.4f} s"
        }
    }

    # 🔹 Wyświetlenie JSON
    print(json.dumps(output_data, indent=4))


def parse_arguments():
    """Obsługa argumentów wiersza poleceń"""
    parser = argparse.ArgumentParser(description="Wykrywanie i OCR tablic rejestracyjnych.")
    parser.add_argument("image_path", type=str, help="Ścieżka do pliku obrazu")
    return parser.parse_args()


def main():
    """Główna funkcja"""
    args = parse_arguments()

    # 🔹 Załaduj modele raz i przekazuj do funkcji
    global yolo_model, parseq_model
    yolo_model = load_yolo_model()
    parseq_model = load_ocr_model()

    process_single_image(args.image_path, yolo_model, parseq_model)


if __name__ == "__main__":
    main()
