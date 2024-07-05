import torch
from ultralytics import YOLO

model = YOLO('yolov5n.yaml')

print("Czy CUDA jest dostępne:", torch.cuda.is_available())
print("Liczba dostępnych urządzeń CUDA:", torch.cuda.device_count())

if __name__ == '__main__':
    path_config = 'E:/USERS/dominik.roczan/PycharmProjects/computervison-detection-and-blur-ocr-plate-yolo/trained_model/data_set/data.yaml'
    path_config = 'E:/USERS/dominik.roczan/PycharmProjects/computervison-detection-and-blur-ocr-plate-yolo/trained_model/data_set-logo/logo.yaml'

    results = model.train(data=path_config, epochs=31, device='gpu', patience=6)

'''cd E:/USERS/dominik.roczan/PycharmProjects/computervison-detection-and-blur-ocr-plate-yolo'''
'''python training.py'''