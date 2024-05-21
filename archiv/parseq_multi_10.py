import asyncio
import os
import csv
from paddleocr import PaddleOCR

ocr = PaddleOCR()


async def ocr_process(image_path):
    try:
        result = ocr.ocr(image_path, cls=True)
        return result
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu: {image_path}")
        print(f"Powód błędu: {str(e)}")
        return [('error', f'Error processing image: {str(e)}')]


async def run_ocr_tasks():
    input_folder = "input_cropped"
    output_folder = "output"
    output_file = "output.csv"

    # Sprawdź, czy folder wyjściowy istnieje, jeśli nie, utwórz go
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, filename))]

    with open(os.path.join(output_folder, output_file), 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Nazwa zdjęcia', 'Tekst', 'CNF score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_path in sorted(image_paths, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))):
            try:
                results = await ocr_process(image_path)
                if results:
                    for result in results:
                        text = result[0][1][0] if result[0][1] else ''
                        score = result[0][-1][1] if result[0][-1] else ''
                        writer.writerow({'Nazwa zdjęcia': os.path.basename(image_path), 'Tekst': text, 'CNF score': score})
                    print(f"Obraz {os.path.basename(image_path)} przetworzony i zapisany do pliku CSV.")
                else:
                    print(f"Obraz {os.path.basename(image_path)} nie został przetworzony poprawnie.")
            except Exception as e:
                print(f"Błąd podczas przetwarzania obrazu: {image_path}")
                print(f"Powód błędu: {str(e)}")
                writer.writerow({'Nazwa zdjęcia': os.path.basename(image_path), 'Tekst': f'Error processing image: {str(e)}', 'CNF score': ''})


def main():
    asyncio.run(run_ocr_tasks())


if __name__ == "__main__":
    main()
