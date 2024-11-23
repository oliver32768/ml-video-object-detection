import argparse
import os
import glob
import cv2

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", help="Directory to save visualisations to", required=True)
    parser.add_argument("--dataset-dir", help="Directory containing image-label pairs", required=True)
    parser.add_argument("--skip-empty", help="Skip frames not containing sports ball bounding boxes")
    return parser.parse_args()

def draw_yolo_bboxes(frame, label_file, frame_width, frame_height):
    """Draws YOLO bounding boxes on the given frame."""
    with open(label_file, 'r') as f:
        for line in f:
            # YOLO format: <class> <x_center> <y_center> <width> <height>
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
            
            x1 = int((x_center - width / 2) * frame_width)
            y1 = int((y_center - height / 2) * frame_height)
            x2 = int((x_center + width / 2) * frame_width)
            y2 = int((y_center + height / 2) * frame_height)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Class {int(class_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def gen_videos(input_folder, output_folder, fps=30):
    os.makedirs(output_folder, exist_ok=True)

    frame_files = glob.glob(os.path.join(input_folder, "*.png"))

    video_frames = {}
    for frame_file in frame_files:
        base_name = os.path.basename(frame_file).split('.')[0]
        video, frame = map(int, base_name.split('-'))
        video_frames.setdefault(video, []).append((frame, frame_file))

    for video, frames in video_frames.items():
        frames.sort(key=lambda x: x[0])
        sorted_frame_files = [frame_file for _, frame_file in frames]

        first_frame = cv2.imread(sorted_frame_files[0])
        height, width, _ = first_frame.shape

        output_path = os.path.join(output_folder, f"{video}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_file in sorted_frame_files:
            frame = cv2.imread(frame_file)
            print(f'Reading frame {frame_file}')

            label_file = frame_file.replace('.png', '.txt')
            if os.path.isfile(label_file):
                print(f'Reading label file {label_file}')
                draw_yolo_bboxes(frame, label_file, width, height)

            video_writer.write(frame)

        video_writer.release()
        print(f"Created video: {output_path}")

def main():
    args = parse_cli_args()
    gen_videos(args.dataset_dir, args.save_dir)

if __name__ == '__main__':
    main()