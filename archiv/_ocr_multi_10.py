import torch
import os
import csv
from PIL import Image
from strhub.data.module import SceneTextDataModule

# Load model and image transforms
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

# Ścieżka do folderu ze zdjęciami
folder_path = 'D:/Machine_Learning/Projekty/02_ObjectDetection-BlurCarLicensePlates/Blur-CarLicensePlates/output_cropped'

# Ścieżka do pliku detections.csv
detections_csv_path = 'D:/Machine_Learning/Projekty/02_ObjectDetection-BlurCarLicensePlates/Blur-CarLicensePlates/output_csv/detections.csv'

# Ścieżka do folderu z wynikami CSV
output_csv_folder = 'D:/Machine_Learning/Projekty/02_ObjectDetection-BlurCarLicensePlates/Blur-CarLicensePlates/output_csv'

# Utwórz folder na pliki CSV, jeśli nie istnieje
os.makedirs(output_csv_folder, exist_ok=True)

# Utwórz plik CSV i zapisz nagłówki
csv_file_path = os.path.join(output_csv_folder, 'results.csv')
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Image Name', 'Detected', 'Confidence', 'Cropped Image', 'OCR'])

# Otwieranie pliku detections.csv i odczytanie zawartości
detections = {}
with open(detections_csv_path, newline='') as detections_file:
    csv_reader = csv.reader(detections_file)
    for row in csv_reader:
        image_name, detected, confidence = row
        detections[image_name] = (detected, confidence)

# Iteracja przez wszystkie pliki w folderze output_cropped
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Wczytaj nazwę pliku bez "_Cropped"
        original_filename = filename.replace("_Cropped", "")

        # Sprawdź, czy nazwa pliku występuje w detections.csv
        detected, confidence = detections.get(original_filename, (None, None))

        # Jeśli obrazek został wykryty, ale nie ma w folderze output_cropped
        if detected and not original_filename in os.listdir(folder_path):
            detected = None
            confidence = None

        # Wczytaj obrazek
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('RGB')

        # Preprocessing
        img_tensor = img_transform(img).unsqueeze(0)

        # Predykcja
        logits = parseq(img_tensor)
        pred = logits.softmax(-1)
        label, _ = parseq.tokenizer.decode(pred)

        # Zapisz wyniki do pliku CSV
        with open(csv_file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [filename, detected, confidence, original_filename, label[0] if label else "Brak tablicy"])

        print(f'Image: {filename}, Decoded label = {label[0] if label else "Brak tablicy"}')

print("Zakończono przetwarzanie.")
