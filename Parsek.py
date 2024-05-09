import asyncio
from paddleocr import PaddleOCR

ocr = PaddleOCR()

async def ocr_process(image_path):
    result = ocr.ocr(image_path, cls=True)
    return result

async def run_ocr_tasks():
    image_paths = ["input/101.jpg","input/102.jpg", "input/103.jpg","input/300.jpg",]

    tasks = [ocr_process(path) for path in image_paths]
    results = await asyncio.gather(*tasks)
    for result in results:
        for line in result:
            print(1,line[0][1])

def main():
    asyncio.run(run_ocr_tasks())

if __name__ == "__main__":
    main()
