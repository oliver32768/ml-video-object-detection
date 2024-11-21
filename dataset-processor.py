import os
import argparse
from ultralytics import YOLO
import cv2

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to model checkpoint used for generating labels", required=True)
    parser.add_argument("--video-dir", help="Path to directory containing videos to label", required=True)
    parser.add_argument("--label-dir", help="Path to directory where RGB frames and their labels should be saved", required=True)
    return parser.parse_args()

def gen_pseudo_annotations(model, video_dir, label_dir):
    """Generates pseudo annotations from scratch; this will dump frames from the input videos and generate pseudo-annotations using CLI specified model checkpoint"""
    
    if not os.path.isdir(video_dir):
        print(f"Video directory specified doesn't exist\n({video_dir})")
        exit()
    os.makedirs(label_dir, exist_ok=True)  # contains JPEGs and labels (TXTs)

    THRESH = 0.1
    SPORTSBALL_IDX = 32

    for file in os.listdir(video_dir):
        filename = os.fsdecode(file)

        filename_ext_removed = filename.split('.')[0]
        print(f'Processing file {filename_ext_removed}')
        cap = cv2.VideoCapture(os.path.join(video_dir, filename))
        frame_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            print(f'Processing frame {frame_idx} in file {filename_ext_removed}')

            # dump RGB frame
            image_path = os.path.join(label_dir, f"{filename_ext_removed}-{frame_idx:05d}.png")
            H,W = frame.shape[:2]
            cv2.imwrite(image_path, frame)

            label_path = os.path.join(label_dir, f"{filename_ext_removed}-{frame_idx:05d}.txt")

            results = model.predict(frame, imgsz=640) # directly use the YOLO predict method

            for box in results[0].boxes:
                cls = int(box.cls.item())
                print(f'class {cls} @ {box.conf.item():.4f} conf.')
                if cls == SPORTSBALL_IDX and box.conf > THRESH:
                    x,y,w,h = box.xywh[0]
                    x,w = x/W, w/W
                    y,h = y/H, h/H
                    with open(label_path, "w") as label_file:
                        label_str = f'{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n'
                        print(f'Writing line to {label_path}: {label_str}')
                        label_file.write(label_str)
            
            frame_idx += 1
    
    cap.release()

def main():
    args = parse_cli_args()
    gen_pseudo_annotations(YOLO(args.model), args.video_dir, args.label_dir)

if __name__ == '__main__':
    main()
