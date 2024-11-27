import argparse
import os
import json
import cv2

def get_id_to_annotation_mapping(annotations):
    id_to_anns = dict()
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in id_to_anns:
            id_to_anns[img_id] = []
        id_to_anns[img_id].append(ann)
    return id_to_anns

def gen_visualisations(dataset_dir):
    annotations_json = os.path.join(dataset_dir, 'annotations.json')
    
    images_dir = os.path.join(dataset_dir, 'images')

    vis_dir = os.path.join(dataset_dir, 'visualisations')
    os.makedirs(vis_dir, exist_ok=True)

    with open(annotations_json, 'r') as f:
        print(f'Reading annotations.json from dataset to be visualised: {annotations_json}')
        annotations = json.load(f)
    
    id_to_anns = get_id_to_annotation_mapping(annotations)

    for img_entry in annotations['images']:
        img_path = os.path.join(images_dir, img_entry['file_name'])
        vis_path = os.path.join(vis_dir, img_entry['file_name'])

        img_id = img_entry['id']

        frame = cv2.imread(img_path)
        img_annotations = id_to_anns[img_id]

        for ann in img_annotations:
            x1, y1, x2, y2 = map(int, ann['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imwrite(vis_path, frame)

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", help="Path to dataset containing images/ folder and annotations.json", required=True)
    return parser.parse_args()

def main():
    args = parse_cli_args()

    gen_visualisations(args.dataset_dir)

if __name__ == '__main__':
    main()