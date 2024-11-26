import argparse
import json
import os
import numpy as np
import shutil

def compute_iou(bbox_a, bbox_b):
    x1_a, y1_a, x2_a, y2_a = bbox_a
    x1_b, y1_b, x2_b, y2_b = bbox_b
    
    # normalise coordinates since FasterRCNN doesn't guarantee spatial order of (x1,y1) vs. (x2,y2)
    x1_a, x2_a = min(x1_a, x2_a), max(x1_a, x2_a)
    y1_a, y2_a = min(y1_a, y2_a), max(y1_a, y2_a)
    x1_b, x2_b = min(x1_b, x2_b), max(x1_b, x2_b)
    y1_b, y2_b = min(y1_b, y2_b), max(y1_b, y2_b)
    
    # intersection area
    inter_x1, inter_x2 = max(x1_a, x1_b), min(x2_a, x2_b)
    inter_y1, inter_y2 = max(y1_a, y1_b), min(y2_a, y2_b)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    
    # union area
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = area_a + area_b - inter_area # avoid double counting intersection
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def prune_doubled_bounding_boxes(annotations):
    cleaned_annotations = []
    
    img_to_anns = {} # {0: [{image_id, bbox, category_id}, ...], 1: ...}
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    for img in annotations['images']:
        num_bboxes = len(img_to_anns[img['id']])
        if num_bboxes > 1: # more than 1 bbox in this image
            for i in range(num_bboxes):
                max_iou = 0.0
                for j in range(i+1,num_bboxes):
                    iou = compute_iou(img_to_anns[img['id']][i]['bbox'], img_to_anns[img['id']][j]['bbox'])
                    max_iou = iou if iou > max_iou else max_iou
                if max_iou < 0.5: # bbox[i] did not have an IoU > 0.5 for any other bbox in this frame - we'll keep it
                    cleaned_annotations.append(img_to_anns[img['id']][i]) # TODO: This will delete all overlapping bounding boxes, but I want to retain one usually
    
    return cleaned_annotations

def prune_narrow_bounding_boxes(annotations):
    cleaned_annotations = []
    for annotation in annotations['annotations']:
        x1, y1, x2, y2 = annotation['bbox']
        w = np.abs(x2 - x1)
        h = np.abs(y2 - y1)
        aspect = w / h if w > h else h / w
        if aspect < 2.0: # TODO: Check this visually makes sense, these are arbitrary numbers for now
            cleaned_annotations.append(annotation)
    return cleaned_annotations

def get_retained_imgs(annotations):
    retained_imgs = []
    retained_ids = set()
    for idx, ann in enumerate(annotations['annotations']):
        respective_img = annotations['images'][ann['image_id']].copy()
        respective_img['id'] = idx # overwriting ids so there aren't any gaps
        
        if ann['image_id'] not in retained_ids: # I haven't already added this image to retained_imgs
            retained_imgs.append(respective_img)
            retained_ids.add(ann['image_id']) # aka 'I retained what was originally image_id = ann['image_id']'

        ann['image_id'] = idx # update this annotation to point at new id
    return retained_imgs

def merge_datasets(prev_annotations, cleaned_annotations, prev_dataset_dir, cur_dataset_dir, cleaned_dataset_dir):
    os.makedirs(os.path.join(cleaned_dataset_dir, 'images'), exist_ok=True)

    cleaned_imgs = set()
    for img in cleaned_annotations['images']:
        cleaned_imgs.add(img['file_name'])

    for img in prev_annotations['images']:
        img_name = img['file_name']
        if img_name not in cleaned_imgs:
            img_path = os.path.join(prev_dataset_dir, img_name)
            # copy from prev. dataset dataset
            shutil.copy(img_path, os.path.join(cleaned_dataset_dir, 'images', img_name))

    for img in cleaned_imgs:
        img_name = img['file_name']
        img_path = os.path.join(cur_dataset_dir, img_name)
        # copy from cur. dataset
        shutil.copy(img_path, os.path.join(cleaned_dataset_dir, 'images', img_name))

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dataset-dir", help="Path to save new dataset to", required=True)
    parser.add_argument("--input-dataset-dir", help="Path to dataset which should be cleaned", required=True)
    parser.add_argument("--prev-dataset-dir", help="Path to dataset to compare against", required=True)
    return parser.parse_args()

def main():
    args = parse_cli_args()

    with open(os.path.join(args.input_dataset_dir, 'annotations.json'), 'r') as f:
        input_annotations = json.load(f)
    with open(os.path.join(args.prev_dataset_dir, 'annotations.json'), 'r') as f:
        prev_annotations = json.load(f)

    # iterate over new dataset
    # delete everything that is spurious
    # create new dataset with:
    # + detections in old that aren't in new
    # + detections in new that aren't in old
    # + for detections in both new and old, prefer new?

    input_annotations['annotations'] = prune_doubled_bounding_boxes(input_annotations)
    input_annotations['annotations'] = prune_narrow_bounding_boxes(input_annotations)
    input_annotations['images'] = get_retained_imgs(input_annotations)

    merge_datasets(prev_annotations, input_annotations, args.prev_dataset_dir, args.input_dataset_dir, args.output_dataset_dir)
    # TODO: add a function (in another file probably) that takes a set of images and the annotations JSON and produces visualisations 
    # - like in dataset-processor.py, just not at inference time

if __name__ == '__main__':
    main()