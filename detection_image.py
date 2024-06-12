import os
from ultralytics import YOLO
import cv2
import csv

# Ścieżka do katalogu zawierającego zdjęcia
directory_path = '02_input_blur'

# Lista na ścieżki do zdjęć
image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.jpg') or f.endswith('.png')]

# Załaduj model YOLO
model_path = 'trained_model/train23/weights/90_best.pt'
model = YOLO(model_path)  # Używając ścieżki do wcześniej wytrenowanego modelu

# Ścieżki do zapisania plików
output_dirs = {
    'blurred': '03_output_blur',
    'cropped': '03_output_cropped',
    'csv': '03_output_csv'
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Ścieżka do pliku CSV
csv_file_path = os.path.join(output_dirs['csv'], 'detections.csv')

# Sprawdzenie, czy plik CSV istnieje; jeśli nie, utwórz nowy plik CSV i zapisz nagłówki
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image name', 'Detection', 'Score'])  # Nagłówki kolumn

# Słownik do śledzenia liczby detekcji i wyników dla każdego pliku
results_dict = {}

# Przetwarzaj wszystkie obrazy w katalogu
for image_path in image_paths:
    image = cv2.imread(image_path)

    # Przeprowadź detekcję na obrazie
    results = model(image)

    detected = False  # Flaga, czy wykryto jakieś tablice
    confidence_scores = []  # Lista na wartości confidence

    # Przetwarzanie wyników detekcji
    for r in results:
        if hasattr(r, 'boxes'):
            boxes = r.boxes
            for b in boxes:
                xyxy = b.xyxy.cpu().numpy().astype(int).flatten()
                confidence = float(b.conf)
                class_id = int(b.cls)
                class_name = r.names[class_id]
                label = f"{class_name} {confidence:.2f}"

                if confidence > 0.15:
                    detected = True
                    confidence_scores.append(confidence)

                    cropped_image = image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    cropped_name = f'{os.path.splitext(os.path.basename(image_path))[0]}_Cropped_{len(confidence_scores)}.jpg'
                    cv2.imwrite(os.path.join(output_dirs['cropped'], cropped_name), cropped_image)

                    cropped_region = image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    blurred_region = cv2.blur(cropped_region, (23, 23))
                    image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = blurred_region

    # Zapisz zmodyfikowane zdjęcie do odpowiedniego folderu
    if detected:
        final_image_path = os.path.join(output_dirs['blurred'], f'{os.path.splitext(os.path.basename(image_path))[0]}_Blur.jpg')
        cv2.imwrite(final_image_path, image)
    else:
        final_image_path = os.path.join(output_dirs['blurred'], os.path.basename(image_path))
        cv2.imwrite(final_image_path, image)
        print(f"No license plates were detected in {image_path}.")

    # Zapisz wyniki detekcji w słowniku
    results_dict[image_path] = confidence_scores if detected else 'None'

# Zapisywanie informacji do pliku CSV
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        detection_count = len(results_dict[image_path]) if results_dict[image_path] != 'None' else 'None'
        scores = ', '.join([f"{score:.2f}" for score in results_dict[image_path]]) if detection_count != 'None' else 'None'
        writer.writerow([filename, detection_count, scores])
