import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np

model = models.efficientnet_b0(pretrained=True)
model.eval()

def detect_deepfake(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

    return "Fake" if torch.argmax(output) == 1 else "Real"