import asyncio
from paddleocr import PaddleOCR

ocr = PaddleOCR()


async def ocr_process(image_path):
    result = ocr.ocr(image_path, cls=True)
    return result


async def run_ocr_tasks():
    image_paths = ["output_canny/21_p_ev_proxy_cropped_1_edges.jpg"]

    tasks = [ocr_process(path) for path in image_paths]
    results = await asyncio.gather(*tasks)
    # count = 0
    for result in results:
        # count += 1
        for line in result:
            # print( line[0][1])
            print( line)


def main():
    asyncio.run(run_ocr_tasks())


if __name__ == "__main__":
    main()
