"""
Use a FasterRCNN v1 model to generate sports-ball bounding box labels (PASCAL VOC) for a set of videos

Usage:
    python dataset-processor.py [options]

Options:
    --video-dir <string>    Path to directory containing MP4 videos to be labelled by model
    --dataset-dir <string>  Path to directory where labels and images should be saved
    --weights <string>      (Optional) Path to FasterRCNN v1 checkpoint. If left unspecified, defaults to downloading pretrained weights
    --nth-frame <int>       (Optional) Process every nth frame only. If left unspecified, every frame is processed
"""

import cv2
import json
import torch
import os
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import argparse

class DatasetProcessor:
    def __init__(self, video_dir, dataset_dir, custom_weights):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if custom_weights is not None:
            print(f'Loading FasterRCNN with custom weights: {os.fsdecode(custom_weights)}')
            self.model = fasterrcnn_resnet50_fpn(weights=None)

            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            num_classes = 2 # Background + 1 class
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            ckpt = torch.load(custom_weights)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()

            self.ball_class_ids = [1] # I change the index to just be 1 during finetuning
        else:
            print(f'Loading FasterRCNN v1 with official PyTorch pre-trained weights')
            self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device)
            self.model.eval()

            self.ball_class_ids = [37] # Have to use the COCO index
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.confidence_threshold = 0.7

        self.images = []
        self.annotations = []
        self.categories = [{'id': 1, 'name': 'sports ball'}]

        self.img_id = 0

        self.video_dir = video_dir
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        print(f'Results will be saved to {self.dataset_dir}')

    def process_frame(self, frame):
        image = F.to_tensor(frame)
        image = image.to(self.device)

        predictions = self.model([image])[0]

        detections = []
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            if label.item() in self.ball_class_ids and score.item() > self.confidence_threshold:
                detection = {
                    'image_id': self.img_id,
                    'bbox': box.cpu().numpy().tolist(),
                    'category_id': 1
                }
                detections.append(detection)
                self.annotations.append(detection)

        return detections

    def process_video(self, video_path, nth_frame=1):
        video_name = Path(video_path).stem

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        with torch.no_grad():
            with tqdm(total=total_frames // nth_frame, desc=f"Processing {video_name}") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % nth_frame != 0:
                        frame_idx += 1
                        continue

                    H, W = frame.shape[:2]
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    detections = self.process_frame(frame_rgb)
                    
                    if len(detections) > 0:
                        frame_filename = f"{video_name}-{frame_idx:06d}.jpg"
                        self.save_frame(frame, frame_filename)
                        self.images.append({
                            'id': self.img_id,
                            'file_name': frame_filename,
                            'width': W,
                            'height': H
                        })
                        self.img_id += 1
                    
                    frame_idx += 1
                    pbar.update(1)
        
        cap.release()

    def save_frame(self, frame, frame_filename):
        frame_path = os.path.join(self.images_dir, frame_filename)
        cv2.imwrite(frame_path, frame)        

    def process_video_folder(self, nth):
        video_paths = list(Path(self.video_dir).glob('*.mp4'))

        for video_path in video_paths:
            print(f"\nProcessing video: {video_path.name}")

            self.process_video(str(video_path), nth)

        data = {
            "images": datasetprocessor.images,
            "annotations": datasetprocessor.annotations,
            "categories": datasetprocessor.categories
        }

        with open(os.path.join(self.dataset_dir, "annotations.json"), "w") as json_file:
            json.dump(data, json_file, indent=4)

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", help="Path to directory containing source videos to be labelled by model", required=True)
    parser.add_argument("--dataset-dir", help="Path to directory where labels and images should be saved", required=True)
    parser.add_argument("--weights", help="Path to model weights (FasterRCNN v1). Defaults to downloading pretrained weights")
    parser.add_argument("--nth-frame", help="Process every nth frame only (specify n as argument)", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_cli_args()
    video_dir = args.video_dir
    dataset_dir = args.dataset_dir
    weights = args.weights
    nth = args.nth_frame if args.nth_frame is not None else 1
    
    datasetprocessor = DatasetProcessor(video_dir, dataset_dir, weights)
    datasetprocessor.process_video_folder(nth)