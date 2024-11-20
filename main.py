import cv2
import torch
from PIL import Image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True)
    model.to(device)

if __name__ == '__main__':
    main()