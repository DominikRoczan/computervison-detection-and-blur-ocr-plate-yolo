import torch
import os
import csv
from PIL import Image
from strhub.data.module import SceneTextDataModule

# Load model and image transforms
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

# Ścieżka do folderu ze zdjęciami
folder_path = 'E:/USERS/dominik.roczan/PycharmProjects/computervison-detection-and-blur-ocr-plate-yolo/03_output_cropped'

# Ścieżka do pliku CSV z wynikami detekcji
detections_csv_path = 'E:/USERS/dominik.roczan/PycharmProjects/computervison-detection-and-blur-ocr-plate-yolo/03_output_csv/detections.csv'

# Wczytaj dane z istniejącego pliku CSV
with open(detections_csv_path, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    headers = next(csv_reader)
    data = list(csv_reader)

# Dodaj nową kolumnę do nagłówków, jeśli jej nie ma
if 'Decoded Text' not in headers:
    headers.append('Decoded Text')

# Iteracja przez wszystkie pliki w folderze i wykonanie predykcji
results = {}
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        try:
            # Wczytaj obrazek
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')

            # Preprocessing
            img = img_transform(img).unsqueeze(0)

            # Predykcja
            with torch.no_grad():
                logits = parseq(img)
            pred = logits.softmax(-1)
            label, _ = parseq.tokenizer.decode(pred)

            # Dodaj wynik do słownika, używając pełnej nazwy pliku
            results[filename] = label[0]

            print(f'Image: {filename}, Decoded label = {label[0]}')

        except Exception as e:
            print(f'Error processing file {filename}: {e}')

# Debugowanie - wyświetlanie wyników OCR
print("OCR Results:")
for k, v in results.items():
    print(f'{k}: {v}')

# Uzupełnianie kolumny "Decoded Text" w danych CSV
updated_data = []
for row in data:
    image_name = row[0]  # Pobierz nazwę pliku z pliku CSV

    decoded_text = results.get(image_name)

    # Debugowanie - sprawdzanie dopasowania nazw plików
    if decoded_text is not None:
        print(f'Matching {image_name} with decoded text {decoded_text}')
    else:
        print(f'No match found for {image_name}')
        decoded_text = 'None'  # Jeśli nie ma dopasowania, ustawiamy 'None'

    if 'Decoded Text' in headers:
        if len(row) < len(headers):  # Jeśli kolumna "Decoded Text" nie istnieje jeszcze w tym wierszu
            row.append(decoded_text)
        else:
            row[headers.index('Decoded Text')] = decoded_text
    else:
        row.append(decoded_text)

    updated_data.append(row)

# Zapisz zaktualizowane dane do pliku CSV
with open(detections_csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(headers)
    csv_writer.writerows(updated_data)
