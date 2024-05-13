import os
from PIL import Image

# Ścieżka do katalogu zawierającego zdjęcia w formacie WEBP
input_directory = 'webp_input'

# Ścieżka, gdzie mają być zapisane zdjęcia w formacie JPG
output_directory = 'input'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Przechodzi przez wszystkie pliki w katalogu
for index, filename in enumerate(sorted(os.listdir(input_directory))):
    if filename.endswith('.webp'):
        # Tworzy pełną ścieżkę do pliku
        file_path = os.path.join(input_directory, filename)

        # Otwiera obraz w formacie WEBP
        image = Image.open(file_path)

        # Konwertuje obraz na format JPG
        rgb_image = image.convert('RGB')

        # Tworzy nową nazwę pliku z odpowiednią numeracją i oryginalną nazwą
        new_filename = '{:02d}_{}.jpg'.format(index + 1, os.path.splitext(filename)[0])
        new_file_path = os.path.join(output_directory, new_filename)

        # Zapisuje obraz w formacie JPG
        rgb_image.save(new_file_path, 'JPEG', quality=90)

print("Konwersja zakończona.")
