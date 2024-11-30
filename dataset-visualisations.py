import argparse
import os
import json
import cv2
from tqdm import tqdm

def get_id_to_annotation_mapping(annotations):
    id_to_anns = dict()
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in id_to_anns:
            id_to_anns[img_id] = []
        id_to_anns[img_id].append(ann)
    return id_to_anns

def new_vid_handle(vis_dir, vid_name, H, W):
    vid_path = os.path.join(vis_dir, 'videos', f'{vid_name}.mp4')
    print(f'\nCreating new video writer for {vid_path}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(vid_path, fourcc, 30, (W, H))

    return video_writer, vid_path

def gen_visualisations(dataset_dir):
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

            if vid_name != cur_vid_name: # Doing this down here so I can use frame dimensions for video writer
                if cur_vid_name is not None:
                    # Write video to visualisations/videos/[cur_vid_name].mp4
                    video_writer.release()
                    print(f"\nCreated video: {vid_path}")
                # Start new video 
                H, W, _ = frame.shape
                video_writer, vid_path = new_vid_handle(vis_dir, vid_name, H, W)
                cur_vid_name = vid_name

            img_annotations = id_to_anns[img_id]

            for ann in img_annotations:
                x1, y1, x2, y2 = map(int, ann['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            cv2.imwrite(vis_path, frame) # Save each frame for easier debugging
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