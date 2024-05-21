import torch
import os
import pdb
from PIL import Image
from strhub.data.module import SceneTextDataModule


# pdb.set_trace()
print(os.getcwd())


# Load model and image transforms
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

img = Image.open('D:/Machine_Learning/Projekty/02_ObjectDetection-BlurCarLicensePlates/'
                 'Blur-CarLicensePlates/02_input_blur/02_image (13).jpg').convert('RGB')
# Preprocess. Model expects a batch of images with shape: (B, C, H, W)
img = img_transform(img).unsqueeze(0)

logits = parseq(img)
logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

# Greedy decoding
pred = logits.softmax(-1)
label, confidence = parseq.tokenizer.decode(pred)
print('Decoded label = {}'.format(label))
print(label, confidence, pred)