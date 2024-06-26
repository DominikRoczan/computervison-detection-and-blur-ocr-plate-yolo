import torch
import os
import csv
from PIL import Image
from strhub.data.module import SceneTextDataModule
import xlsxwriter

# Load model and image transforms
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

# Ścieżka do folderu ze zdjęciami
# folder_path = 'E:/USERS/dominik.roczan/PycharmProjects/computervison-detection-and-blur-ocr-plate-yolo/03_output_cropped'
folder_path = '../03_output_cropped'

# Ścieżka do pliku CSV z wynikami detekcji
# detections_csv_path = 'E:/USERS/dominik.roczan/PycharmProjects/computervison-detection-and-blur-ocr-plate-yolo/03_output_csv/detections.csv'
detections_csv_path = '../03_output_csv/detections.csv'

# Wczytaj dane z istniejącego pliku CSV
with open(detections_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    headers = next(csv_reader)
    data = list(csv_reader)

# Dodaj nową kolumnę do nagłówków, jeśli jej nie ma
if 'Decoded Text' not in headers:
    headers.append('Decoded Text')

# Dodaj nową kolumnę do nagłówków, jeśli jej nie ma
if 'Hyperlink' not in headers:
    headers.append('Hyperlink')

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

# Uzupełnianie kolumny "Decoded Text" i "Hyperlink" w danych CSV
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
        if len(row) < len(headers) - 1:  # Jeśli kolumna "Decoded Text" nie istnieje jeszcze w tym wierszu
            row.append(decoded_text)
        else:
            row[headers.index('Decoded Text')] = decoded_text
    else:
        row.append(decoded_text)

    # Dodanie hiperłącza do pliku
    file_path = os.path.join(folder_path, image_name).replace("\\", "/")
    file_link = f'file:///{file_path}'
    if 'Hyperlink' in headers:
        if len(row) < len(headers):  # Jeśli kolumna "Hyperlink" nie istnieje jeszcze w tym wierszu
            row.append(file_link)
        else:
            row[headers.index('Hyperlink')] = file_link
    else:
        row.append(file_link)

    updated_data.append(row)

# Zapisz zaktualizowane dane do pliku CSV
with open(detections_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(headers)
    csv_writer.writerows(updated_data)

# Zapisz zaktualizowane dane do pliku Excel
# excel_file = 'E:/USERS/dominik.roczan/PycharmProjects/computervison-detection-and-blur-ocr-plate-yolo/03_output_csv/detections.xlsx'
excel_file = '../03_output_csv/detections.xlsx'
workbook = xlsxwriter.Workbook(excel_file)
worksheet = workbook.add_worksheet()

# Nagłówki
for col, header in enumerate(headers):
    worksheet.write(0, col, header)

# Dane
row = 1
for data_row in updated_data:
    for col, cell_value in enumerate(data_row):
        if col == headers.index('Hyperlink'):
            worksheet.write_url(row, col, cell_value, string='Open Image')  # Dodanie hiperłącza
        else:
            worksheet.write(row, col, cell_value)
    row += 1

workbook.close()
print(f"Wyniki zapisane w pliku {excel_file}")

# Otwarcie folderu zawierającego plik Excel po zakończeniu
os.startfile(os.path.dirname(os.path.abspath(excel_file)))
