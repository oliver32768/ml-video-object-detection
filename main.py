"""
Fine-tune a FasterRCNN v1 model on the pseudo-labelled dataset

Usage:
    python main.py [options]

Options:
    --dataset-dir <string>  Path to directory containing 'images' folder and 'annotations.json'
    --tag <string>          Identifier to append to the names of checkpoints saved
    --num-epochs <int>      Number of epochs to train for
    --weights <string>      (Optional) Path to FasterRCNN v1 checkpoint. If left unspecified, defaults to downloading pretrained weights
    --no-transform          (Optional) Disables data augmentations
"""

import json
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from torchvision.ops import box_iou
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import argparse
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
        img_info = self.coco['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

            image = F.to_tensor(transformed['image'])
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            image = F.to_tensor(image)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        return image, target

def get_transform():
    transform = A.Compose([
        # FasterRCNN v1 does its own normalisation via ImageNet statistics internally so I leave it out here
        A.RandomSizedBBoxSafeCrop(width=800, height=800),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    return transform

def get_model(num_classes, freeze_backbone, tune_rpn, weights):
    # pretrained fasterrcnn
    if weights is None:
        # load pretrained weights
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        # replace classifier to only classify background and sports balls
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        # don't load pretrained weights
        model = fasterrcnn_resnet50_fpn(weights=None) 

        # still need to replace classifier in order to match architecture of saved weights
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt['model_state_dict'])
    
    # freeze resnet backbone
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
            
    # freeze region proposal network
    if not tune_rpn:
        for param in model.rpn.parameters():
            param.requires_grad = False
    
    # configure anchors to prefer near-square shapes for sports balls
    model.rpn.anchor_generator.sizes = ((32, 64, 128, 256),) * 5
    model.rpn.anchor_generator.aspect_ratios = ((0.8, 1.0, 1.2),) * 5
    
    return model

def compute_metrics(all_gt_boxes, all_pred_boxes, iou_threshold):
    tp = fp = fn = 0
    iou_scores = []
    
    for gt_boxes, pred_boxes in zip(all_gt_boxes, all_pred_boxes):
        if pred_boxes.numel() == 0:
            fn += len(gt_boxes)
            continue
        
        ious = box_iou(gt_boxes, pred_boxes)
        max_iou, _ = ious.max(dim=1)
        
        tp += (max_iou >= iou_threshold).sum().item()
        fn += (max_iou < iou_threshold).sum().item()
        
        max_iou_pred, _ = ious.max(dim=0)
        fp += (max_iou_pred < iou_threshold).sum().item()
        
        iou_scores.extend(max_iou.numpy())
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    avg_iou = np.mean(iou_scores) if iou_scores else 0.0
    
    return precision, recall, f1_score, avg_iou

def validation(model, data_loader, epoch, device):
    print(f'Beginning validation for epoch {epoch}...')
    model.eval()
    
    total_loss = 0
    num_batches = 0
    all_gt_boxes = []
    all_pred_boxes = []
    with tqdm(data_loader, unit="batch") as tqdm_dl:
        for images, targets in tqdm_dl:
            tqdm_dl.set_description(f"(Val.) Epoch {epoch + 1}")

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            with torch.no_grad():
                outputs = model(images)

                for i in range(len(images)):
                    gt_boxes = targets[i]['boxes'].cpu()
                    pred_boxes = outputs[i]['boxes'].cpu()
                    all_gt_boxes.append(gt_boxes)
                    all_pred_boxes.append(pred_boxes)

                model.train() # you have to do this unfortunately
                loss_dict = model(images, targets)
                model.eval()

                losses = sum(loss for loss in loss_dict.values())

                total_loss += losses.item()
                num_batches += 1

                avg_loss = total_loss / num_batches
                tqdm_dl.set_postfix(avg_loss=avg_loss)

    precision, recall, f1_score, avg_iou = compute_metrics(all_gt_boxes, all_pred_boxes, iou_threshold=0.5)

    model.train()
    
    print(f'(Epoch {epoch}) Val. Metrics: Loss = {avg_loss:.4f} Precision = {precision:.4f} Recall = {recall:.4f} F1 = {f1_score:.4f} Avg. IoU = {avg_iou:.4f}')    
    return total_loss / len(data_loader)

def train_one_epoch(model, optimizer, data_loader, epoch, device):
    model.train()
    
    total_loss = 0
    num_batches = 0
    with tqdm(data_loader, unit="batch") as tqdm_dl:
        for images, targets in tqdm_dl:
            tqdm_dl.set_description(f"(Train) Epoch {epoch + 1}")

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
            tqdm_dl.set_postfix(avg_loss=avg_loss)
        
    return total_loss / len(data_loader)

def save_ckpt(epoch, model_state, optimizer_state, loss, name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'loss': loss,
    }, os.path.join('models', f'{name}.pth'))

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", help="Path to directory containing images/ folder and annotations.json", required=True)
    parser.add_argument("--tag", help="Identifier to append to the names of checkpoints saved", required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--weights", help="Path to model weights (FasterRCNN v1). Defaults to downloading pretrained weights")
    parser.add_argument("--no-transform", action="store_true")
    return parser.parse_args()

def main():
    args = parse_cli_args()

    # Initialisation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.no_transform:
        transforms = None
    else:
        transforms = get_transform()
    
    train_dataset = SportsballDataset(
        json_file=os.path.join(args.dataset_dir, 'annotations.json'),
        img_dir=os.path.join(args.dataset_dir, 'images'),
        transform=transforms
    )

    dataset_size = len(train_dataset)

    split_ratio = 0.8 # amount of original dataset to reserve for training, the rest is for validation
    train_size = int(split_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    freeze_backbone=True
    num_classes = len(train_dataset.coco['categories']) + 1
    model = get_model(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        tune_rpn=False,
        weights=args.weights
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
    head_params = {"params": [p for n, p in model.roi_heads.named_parameters() if p.requires_grad], "lr": lr}
    params.append(head_params)
    
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    os.makedirs('models', exist_ok=True)
    
    # Training loop
    min_loss = 1e+10
    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, epoch, device)
        val_loss = validation(model, val_loader, epoch, device)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss:.4f} Val. Loss = {val_loss:.4f}")
        
        if val_loss < min_loss:
            # deliberately not overwriting last if it's the best epoch
            save_ckpt(epoch, model.state_dict(), optimizer.state_dict(), loss, f'checkpoint_best-{args.tag}')
            min_loss = val_loss
        else:
            save_ckpt(epoch, model.state_dict(), optimizer.state_dict(), loss, f'checkpoint_last-{args.tag}')

if __name__ == "__main__":
    main()