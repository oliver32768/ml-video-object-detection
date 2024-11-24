import cv2
import json
import torch
import os
from pathlib import Path
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

class DatasetProcessor:
    def __init__(self, video_dir, dataset_dir, custom_weights):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if custom_weights is not None:
            # The first pass is done using v1, which trains a v2, which is then used for the future passes
            print(f'Loading FasterRCNN v2 with custom weights: {os.fsdecode(custom_weights)}')
            self.model = fasterrcnn_resnet50_fpn(weights=None)

            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            num_classes = 2 # Background + 1 class
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            ckpt = torch.load(custom_weights)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model = self.model.to(self.device).eval()

            self.ball_class_ids = [1] # I change the index to just be 1 during finetuning
        else:
            # use v1 for first pseudo-labelling pass since it produces a lot less false positives (and I only train on labelled frames)
            print(f'Loading FasterRCNN v1 with official PyTorch pre-trained weights')
            self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(self.device).eval()

            self.ball_class_ids = [37] # Have to use the COCO index
        
        self.confidence_threshold = 0.7

        self.images = []
        self.annotations = []
        self.categories = [{'id': 1, 'name': 'sports ball'}]

        self.img_id = 0

        self.video_dir = video_dir
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.vis_dir = os.path.join(self.dataset_dir, 'visualisations')
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        print(f'Results will be saved to {self.dataset_dir}')

    def process_frame(self, frame):
        image = F.to_tensor(frame)
        image = image.to(self.device)

        predictions = self.model([image])[0]

        detections = []
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            print(label)
            if label.item() in self.ball_class_ids and score.item() > self.confidence_threshold:
                detection = {
                    'image_id': self.img_id,
                    'bbox': box.cpu().numpy().tolist(),
                    'category_id': 1
                }
                detections.append(detection)
                self.annotations.append(detection)

        return detections

    def process_video(self, video_path):
        video_name = Path(video_path).stem

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        with torch.no_grad():
            with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    H, W = frame_rgb.shape[:2]
                    
                    detections = self.process_frame(frame_rgb)
                    
                    if len(detections) > 0:
                        frame_filename = f"{video_name}-{frame_idx:06d}.jpg"
                        self.save_frame(frame, frame_filename, detections)
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

    def save_frame(self, frame, frame_filename, detections):
        frame_path = os.path.join(self.images_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        vis_path = os.path.join(self.vis_dir, frame_filename)
        for detect in detections:
            x1, y1, x2, y2 = map(int, detect['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(vis_path, frame)
        

    def process_video_folder(self):
        video_paths = list(Path(self.video_dir).glob('*.mp4'))

        for video_path in video_paths:
            print(f"\nProcessing video: {video_path.name}")

            self.process_video(str(video_path))

        data = {
            "images": datasetprocessor.images,
            "annotations": datasetprocessor.annotations,
            "categories": datasetprocessor.categories
        }

        with open(os.path.join(self.dataset_dir, "annotations.json"), "w") as json_file:
            json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    video_dir = os.path.join("videos", "train")
    dataset_dir = os.path.join("ball_detection_dataset", "1")
    weights = os.path.join("models", "checkpoint_epoch_1.pth")
    
    datasetprocessor = DatasetProcessor(video_dir, dataset_dir, weights)
    datasetprocessor.process_video_folder()