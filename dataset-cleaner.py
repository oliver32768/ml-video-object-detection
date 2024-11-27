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

def get_id_to_annotation_mapping(annotations):
    id_to_anns = dict()
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in id_to_anns:
            id_to_anns[img_id] = []
        id_to_anns[img_id].append(ann)
    return id_to_anns

def prune_overlapping_bounding_boxes(annotations):
    cleaned_annotations = []
    
    id_to_anns = get_id_to_annotation_mapping(annotations) # {0: [{image_id, bbox, category_id}, ...], 1: ...}
    OVERLAP_THRESH = 0.8

    for img in annotations['images']:
        num_bboxes = len(id_to_anns[img['id']])

        if num_bboxes > 1: # more than 1 bbox in this image
            retained_boxes = set()
            for i in range(num_bboxes):
                overlapping_bboxes = []
                overlapping_bboxes.append((id_to_anns[img['id']][i]['bbox'], i))

                for j in range(i+1,num_bboxes):
                    iou = compute_iou(id_to_anns[img['id']][i]['bbox'], id_to_anns[img['id']][j]['bbox'])

                    if iou > OVERLAP_THRESH:
                        # bbox i and j overlap significantly
                        overlapping_bboxes.append((id_to_anns[img['id']][j]['bbox'], j))

                largest_bbox = 0.0
                candidate_idx = 0
                for bbox, idx in overlapping_bboxes:
                    # find largest box amongst the overlapping boxes
                    x1, y1, x2, y2 = bbox
                    area = np.abs(x2-x1) * np.abs(y2-y1)
                    if area > largest_bbox:
                        largest_bbox = area
                        candidate_idx = idx
                
                if candidate_idx not in retained_boxes:
                    # if it hasn't already been added to the new annotation set, add it
                    cleaned_annotations.append(id_to_anns[img['id']][candidate_idx])
                    retained_boxes.add(candidate_idx)
        else:
            cleaned_annotations.append(id_to_anns[img['id']][0])
    
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

def get_retained_imgs(new_annotations):
    # reconstruct the annotations.json knowing which images we've retained after cleaning the dataset
    # specifically, the array of images - some annotations were dropped so now we drop the corresponding images
    # likewise we need to add annotation - image pairs for images being merged from the old dataset

    retained_imgs = []
    retained_ids = set()
    for idx, ann in enumerate(new_annotations['annotations']):
        respective_img = new_annotations['images'][ann['image_id']].copy() # get the image corresponding to this annotation
        respective_img['id'] = idx # overwriting ids so there aren't any gaps
        
        if ann['image_id'] not in retained_ids: 
            # I haven't already added this image to retained_imgs - we need to test this since some images have multiple annotations
            retained_imgs.append(respective_img)
            retained_ids.add(ann['image_id']) # aka 'I retained what was originally image_id = ann['image_id']'

        ann['image_id'] = idx # update this annotation to point at new id

    return retained_imgs

def copy_img(src_dataset_dir, dest_dataset_dir, img_name):
    img_path = os.path.join(src_dataset_dir, 'images', img_name)
    shutil.copy(img_path, os.path.join(dest_dataset_dir, 'images', img_name))

def copy_img_annotation_entries(annotations, merged_img_name, merged_images_list, merged_annotations_list, id_to_anns, idx):
    for prev_img in annotations['images']:
        if prev_img['file_name'] == merged_img_name:
            # then copy over that images[] entry and its respective (by id) annotations[] entries into a new dictionary. use idx as img id
            merged_img_entry = prev_img.copy()
            merged_img_entry['id'] = idx
            merged_images_list.append(merged_img_entry)

            for annotation in id_to_anns[prev_img['id']]:
                merged_annotations_entry = annotation.copy()
                merged_annotations_entry['image_id'] = idx
                merged_annotations_list.append(merged_annotations_entry)

def merge_datasets(prev_annotations, new_annotations, prev_dataset_dir, input_dataset_dir, output_dataset_dir):
    os.makedirs(os.path.join(output_dataset_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dataset_dir, 'visualisations'), exist_ok=True)

    print(f'prev_dataset_dir   : {prev_dataset_dir}')
    print(f'input_dataset_dir  : {input_dataset_dir}')
    print(f'output_dataset_dir : {output_dataset_dir}')

    print(f'Building unordered set of cleaned image names')
    cleaned_img_names = set()
    for img in new_annotations['images']:
        # get the name of every image file left in the cleaned dataset
        cleaned_img_names.add(img['file_name'])

    print(f'Copying all labelled images from original dataset not appearing in new, cleaned dataset into {output_dataset_dir}')
    for img in prev_annotations['images']:
        img_name = img['file_name']
        if img_name not in cleaned_img_names:
            # copy all labelled images from the old dataset which did not appear in the current dataset
            copy_img(prev_dataset_dir, output_dataset_dir, img_name)

    print(f'Copying all labelled images from new, cleaned dataset into {output_dataset_dir}')
    for img_name in cleaned_img_names:
        # copy all labelled images from the new dataset which survived cleaning
        copy_img(input_dataset_dir, output_dataset_dir, img_name)

    prev_id_to_anns = get_id_to_annotation_mapping(prev_annotations)
    new_id_to_anns = get_id_to_annotation_mapping(new_annotations)

    merged_annotations = []
    merged_images = []
    print(f'Building final annotations.json dictionary from merged dataset images')
    for idx, merged_img_name in enumerate(sorted(os.listdir(os.path.join(output_dataset_dir, 'images')))):
        if merged_img_name not in cleaned_img_names: # this file has come from the old dataset
            # look inside prev_annotations for images[] entry with same file name
            copy_img_annotation_entries(prev_annotations, merged_img_name, merged_images, merged_annotations, prev_id_to_anns, idx)
        else:
            # same as above but for new_annotations
            copy_img_annotation_entries(new_annotations, merged_img_name, merged_images, merged_annotations, new_id_to_anns, idx)

    return merged_annotations, merged_images
    # return merged images and annotations lists and write into new dict. copy over old categories[] and then dump to json in output_dataset_dir
        

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dataset-dir", help="Path to save new dataset to", required=True)
    parser.add_argument("--input-dataset-dir", help="Path to dataset which should be cleaned", required=True)
    parser.add_argument("--prev-dataset-dir", help="Path to dataset to compare against", required=True)
    return parser.parse_args()

def main():
    args = parse_cli_args()

    input_json_path = os.path.join(args.input_dataset_dir, 'annotations.json')
    with open(input_json_path, 'r') as f:
        print(f'Reading annotations.json from dataset to be cleaned: {input_json_path}')
        input_annotations = json.load(f)

    old_json_path = os.path.join(args.prev_dataset_dir, 'annotations.json')
    with open(old_json_path, 'r') as f:
        print(f'Reading annotations.json from dataset to be cleaned: {old_json_path}')
        prev_annotations = json.load(f)
    
    new_annotations = input_annotations.copy()

    print(f'Pruning overlapping bounding boxes from input dataset')
    new_annotations['annotations'] = prune_overlapping_bounding_boxes(new_annotations)

    print(f'Pruning overly narrow bounding boxes from input dataset')
    new_annotations['annotations'] = prune_narrow_bounding_boxes(new_annotations)

    print(f'Collating images which survived cleaning and updating dictionary IDs')
    new_annotations['images'] = get_retained_imgs(new_annotations)

    merged_annotations = dict()

    print(f'Merging image files from old and cleaned datasets')
    merged_annotations['annotations'], merged_annotations['images'] = merge_datasets(prev_annotations, new_annotations, args.prev_dataset_dir, args.input_dataset_dir, args.output_dataset_dir)
    merged_annotations['categories'] = prev_annotations['categories']

    output_json_path = os.path.join(args.output_dataset_dir, "annotations.json")
    with open(output_json_path, "w") as json_file:
        print(f'Dumping final dictionary to {output_json_path}')
        json.dump(merged_annotations, json_file, indent=4)

if __name__ == '__main__':
    main()