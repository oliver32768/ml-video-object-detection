import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm

class SportsballDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file, 'r') as f:
            self.coco = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform
        
        # Create image_id to annotations mapping
        self.img_to_anns = {} # {0: [{image_id, bbox, category_id}, ...], 1: ...}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
    def __len__(self):
        return len(self.coco['images'])
    
    def __getitem__(self, idx):
        # Get image info
        img_info = self.coco['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Get annotations
        anns = self.img_to_anns.get(img_info['id'], [])
        
        # Prepare boxes and labels
        boxes = []
        labels = []
        
        for ann in anns:
            x1, y1, x2, y2 = ann['bbox']
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        #boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = np.array(boxes)
        labels = np.array(labels)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])
            
            # Ensure boxes are valid (all coordinates are within image bounds)
            if len(boxes) > 0:
                boxes = np.clip(boxes, 0, 800)  # clip to image size after transforms
            
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            image = F.to_tensor(image)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        return image, target

def get_transform(train):
    if train:
        transform = A.Compose([
            A.RandomSizedBBoxSafeCrop(width=800, height=800),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        transform = A.Compose([
            A.Resize(800, 800),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    return transform

def get_model(num_classes):
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    
    total_loss = 0
    num_batches = 0
    with tqdm(data_loader, unit='batch') as tepoch:
        for images, targets in tepoch:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            num_batches += 1
            avg_loss = total_loss / num_batches
            tepoch.set_postfix(avg_loss=avg_loss)
        
    return total_loss / len(data_loader)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = SportsballDataset(
        json_file=os.path.join('ball_detection_dataset', 'annotations.json'),
        img_dir=os.path.join('ball_detection_dataset', 'images'),
        transform=get_transform(train=True)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize model (num_classes = number of classes + background)
    num_classes = len(train_dataset.coco['categories']) + 1
    model = get_model(num_classes)
    model.to(device)
    
    # Initialize optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Make model checkpoint directory
    os.makedirs('models', exist_ok=True)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join('models', f'checkpoint_epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    main()