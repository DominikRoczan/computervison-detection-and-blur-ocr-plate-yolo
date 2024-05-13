from ultralytics import YOLO
import cv2
import easyocr
import os

from archiv.import_image import image_path  # Zaimportuj ścieżkę do obrazu

# Lista obrazów do przetworzenia
image_paths = image_path
image = cv2.imread(image_paths)

# Załaduj model YOLO
model_path = '../trained_model/weights/best.pt'
model = YOLO(model_path)  # Używając ścieżki do wcześniej wytrenowanego modelu

# Przeprowadź detekcję na listę obrazów
results = model(image_paths)  # Zwraca listę obiektów Results

# Utwórz folder na wyniki, jeśli nie istnieje
dir_name = '../output'
if not os.path.exists(dir_name):
    os.makedirs(dir_name, exist_ok=True)

idx = 0
detected = False  # Flaga, czy wykryto jakieś tablice
cropped_images = []  # Lista na wycięte obrazy tablic
label_endpoints = []  # Lista na punkty końcowe etykiet tablic rejestracyjnych

# Przetwarzanie wyników
for r in results:
    boxes = r.boxes
    class_names = list(r.names.values())

    # Iteruj przez każde pole (bounding box)
    for b in boxes:

        xyxy = b.xyxy  # Zakładamy, że xyxy to tensor z 4 wartościami [x1, y1, x2, y2]
        confidence = float(b.conf)  # Prawidłowa pewność detekcji

        class_id = b.cls.item()  # Get class ID for each box
        class_name = r.names[class_id]  # Convert class ID to class name

        label = f"{class_name} {confidence:.2f}"

        if confidence > 0.15:
            detected = True
            box_width = int(xyxy[0][2]) - int(xyxy[0][0])
            idx += 1

            # Skaluj tekst, aby był 1.5x szerszy niż bounding box
            label_width = box_width * 1.5
            # Początkowy rozmiar czcionki
            font_scale = 0.5
            # Oblicz rozmiar tekstu z początkowym rozmiarem czcionki
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            # Skaluj font_scale do osiągnięcia żądanej szerokości etykiety
            font_scale *= label_width / text_size[0]
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

            # Ustawienie nowych współrzędnych tła tekstu
            text_x = int(xyxy[0][0]) - 1
            text_y = int(xyxy[0][1])
            background_tl = (text_x, text_y - text_size[1])
            background_br = (text_x + text_size[0], text_y)

            # Rysowanie etykiety
            cv2.rectangle(image, (int(xyxy[0][0]), int(xyxy[0][1])), (int(xyxy[0][2]), int(xyxy[0][3])), (0, 0, 255), 2)
            cv2.rectangle(image, background_tl, background_br, (0, 0, 255), cv2.FILLED)
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

            # cropped plates
            cropped_image = image[int(xyxy[0][1]):int(xyxy[0][3]), int(xyxy[0][0]):int(xyxy[0][2])]
            cropped_image_path = os.path.join(dir_name, f'cropped_plate_{idx}.jpg')
            cv2.imwrite(cropped_image_path, cropped_image)

            # Wycięcie i zapisanie tablicy rejestracyjnej
            x1, y1, x2, y2 = int(xyxy[0][0]), int(xyxy[0][1]), int(xyxy[0][2]), int(xyxy[0][3])
            cropped_image = image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)

            # Obliczenie punktu końcowego etykiety na osi pionowej
            label_endpoints.append((text_x + text_size[0], text_y + text_size[1] // 2))

# OCR
ocr = easyocr.Reader(['en'])

# Ścieżka do zapisania pliku tekstowego
ocr_output_path = os.path.join(dir_name, 'ocr_output.txt')

# Otwórz plik do zapisu wyników OCR
with open(ocr_output_path, 'w') as f:
    ocr_results = []  # Lista na wyniki OCR
    for cropped_image in cropped_images:
        # Odczytaj tekst z wyciętego obrazu
        result = ocr.readtext(cropped_image)
        ocr_results.append(result)  # Dodaj wynik do listy

        # Zapisz odczytany tekst do pliku
        f.write('Detected text:\n')
        for detection in result:
            f.write(f'{detection[1]}\n')
        f.write('\n')

ocr_res = []

# Dodanie tekstów pod tablicami na bocznym pasku
for cropped_image, endpoint, result in zip(cropped_images, label_endpoints, ocr_results):
    # Konwertuj wynik OCR na ciąg tekstowy
    detected_text = '\n'.join(detection[1] for detection in result)
    ocr_res.append(detected_text)  # Dodaj przetworzony tekst do listy

if detected:
    # Zapisywanie i wyświetlanie obrazu końcowego
    final_image_path = os.path.join(dir_name, 'detected_plate.jpg')
    cv2.imwrite(final_image_path, image)
    # cv2.imshow('Detected Image with Plates and Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # Obsługa przypadku, gdy nie wykryto tablic
    print("No license plates were detected.")
    final_image_path = os.path.join(dir_name, 'original_image.jpg')
    cv2.imwrite(final_image_path, image)
    # cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
