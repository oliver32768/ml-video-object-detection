"""
Generates bounding box visualisations for every frame in an existing dataset (containing images/*.jpg and annotations.json)
Saves visualisations as individual frames (JPEGs) and videos (MP4s)

Usage:
    python dataset-visualisations.py [options]

Options:
    --dataset-dir <string>  Path to dataset containing images/ folder and annotations.json
"""

import argparse
import os
import json
import cv2
from tqdm import tqdm

def get_id_to_annotation_mapping(annotations):
    # Create image_id to annotations mapping
    # {0: [{image_id, bbox, category_id}, ...], 1: ...}
    # i.e. id -> list of annotation dictionaries (each containing one bounding box)

    id_to_anns = dict()
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in id_to_anns:
            id_to_anns[img_id] = []
        id_to_anns[img_id].append(ann)
    return id_to_anns

def new_vid_handle(vis_dir, vid_name, H, W):
    # Start writing to a new video file

    vid_path = os.path.join(vis_dir, 'videos', f'{vid_name}.mp4')
    print(f'\nCreating new video writer for {vid_path}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(vid_path, fourcc, 30, (W, H))

    return video_writer, vid_path

def gen_visualisations(dataset_dir):
    # Iterate over dataset and save frames with their respective bounding boxes overlaid into JPEGs and MP4s

    annotations_json = os.path.join(dataset_dir, 'annotations.json')
    
    images_dir = os.path.join(dataset_dir, 'images')

    vis_dir = os.path.join(dataset_dir, 'visualisations')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(os.path.join(vis_dir, 'videos'), exist_ok=True)

    with open(annotations_json, 'r') as f:
        print(f'Reading annotations.json from dataset to be visualised: {annotations_json}')
        annotations = json.load(f)
    
    id_to_anns = get_id_to_annotation_mapping(annotations)

    cur_vid_name = None

    with tqdm(annotations['images'], unit="img") as tqdm_imgs:
        for img_entry in tqdm_imgs:
            img_name = img_entry['file_name']
            tqdm_imgs.set_description(f"Image: {img_name}")
            
            vid_name = img_name.split('-')[0]

            img_path = os.path.join(images_dir, img_name)
            vis_path = os.path.join(vis_dir, img_name)

            img_id = img_entry['id']

            frame = cv2.imread(img_path)

            if vid_name != cur_vid_name:
                # annotations.json contains numerically ordered frames as {vid}-{frame}.jpg
                # if {vid} changes, we should write the video (if it exists) and begin a new one

                # if cur_vid_name is None, this is the first video, so there is nothing yet to write
                if cur_vid_name is not None: 
                    video_writer.release()
                    print(f"\nCreated video: {vid_path}")
                
                # Get new video writer
                H, W, _ = frame.shape
                video_writer, vid_path = new_vid_handle(vis_dir, vid_name, H, W)
                cur_vid_name = vid_name

            img_annotations = id_to_anns[img_id]

            for ann in img_annotations:
                # Overlay bounding boxes
                x1, y1, x2, y2 = map(int, ann['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # Save each frame for easier debugging
            cv2.imwrite(vis_path, frame) 
            video_writer.write(frame)

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", help="Path to dataset containing images/ folder and annotations.json", required=True)
    return parser.parse_args()

def main():
    args = parse_cli_args()

    gen_visualisations(args.dataset_dir)

if __name__ == '__main__':
    main()