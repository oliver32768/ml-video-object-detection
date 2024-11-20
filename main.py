import torch
from ultralytics import YOLO

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = YOLO("yolov10n.pt")
    model.to(device)

    print("Model successfully loaded onto:", next(model.model.parameters()).device)

if __name__ == '__main__':
    main()