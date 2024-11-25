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
        img_info = self.coco['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        anns = self.img_to_anns.get(img_info['id'], [])
        
        boxes = []
        labels = []
        
        for ann in anns:
            x1, y1, x2, y2 = ann['bbox']
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        
        boxes = np.array(boxes)
        labels = np.array(labels)
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])
            
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
        # TODO: Motion blur
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

def get_model(num_classes, freeze_backbone=True, tune_rpn=False, pretrained=True):
    # pretrained fasterrcnn (TODO: loading from ckpt)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    # freeze resnet backbone
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
            
    # freeze region proposal network
    if not tune_rpn:
        for param in model.rpn.parameters():
            param.requires_grad = False
    
    # replace classifier to only classify background and sports balls
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # configure anchors to prefer near-square shapes for sports balls
    # AFAIK the model is free to unlearn this, so I should probably add a strict cutoff for extremely low likelihood shapes
    model.rpn.anchor_generator.sizes = ((32, 64, 128, 256),) * 5
    model.rpn.anchor_generator.aspect_ratios = ((0.8, 1.0, 1.2),) * 5
    
    return model

def train_one_epoch(model, optimizer, data_loader, epoch, device):
    model.train()
    
    total_loss = 0
    num_batches = 0
    for images, targets in data_loader:
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
        print(f'(Epoch {epoch}) Batch {num_batches:03d}/{len(data_loader):03d}: Avg. Loss = {avg_loss:.4f}')
        
    return total_loss / len(data_loader)

def main():
    # Initialisation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = SportsballDataset(
        json_file=os.path.join('ball_detection_dataset', 'train', '0', 'annotations.json'),
        img_dir=os.path.join('ball_detection_dataset', 'train', '0', 'images'),
        transform=get_transform(train=True)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    freeze_backbone=True
    num_classes = len(train_dataset.coco['categories']) + 1
    model = get_model(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        tune_rpn=False,
        pretrained=True
    )
    model.to(device)
    
    # model parameter groups
    params = []
    lr = 1e-3
    
    if not freeze_backbone:
        # reduce LR for backbone
        backbone_params = {"params": [p for n, p in model.backbone.named_parameters() if p.requires_grad], "lr": lr / 10}
        params.append(backbone_params)

    # increase LR for new layers
    head_params = {"params": [p for n, p in model.roi_heads.parameters() if p.requires_grad], "lr": lr}
    params.append(head_params)
    
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    os.makedirs('models', exist_ok=True)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, epoch, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join('models', f'checkpoint_epoch_{epoch+1}-v1.pth'))

if __name__ == "__main__":
    main()