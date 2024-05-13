import asyncio
import os
from paddleocr import PaddleOCR

ocr = PaddleOCR()


async def ocr_process(image_path):
    result = ocr.ocr(image_path, cls=True)
    return result


async def run_ocr_tasks():
    # Ścieżka do folderu z obrazami
    input_folder = "input"
    # Pobierz listę plików w folderze
    image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, filename))]

    tasks = [ocr_process(path) for path in image_paths]
    results = await asyncio.gather(*tasks)
    count = 0
    for result in results:
        count += 1
        print(f"Obraz {count}:")
        for line in result:
            # print(line[0][1])
            print(line)



def main():
    asyncio.run(run_ocr_tasks())


if __name__ == "__main__":
    main()
