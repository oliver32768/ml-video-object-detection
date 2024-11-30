"""
Downloads videos from the Sports-1M JSON according to a specified set of label indices

Usage:
    python dataset-downloader.py [options]

Options:
    --target-labels-csv <string>    Path to CSV containing [label-idx],[label-name] entries for Sports-1M labels; restricts videos being downloaded to these labels
    --sports-dataset-json <string>  Path to Sports-1M JSON from which videos should be downloaded
    --output-path <string>          Path to directory to save videos to
"""

import os
import csv
import json
import glob
import yt_dlp
import argparse

def delete_temp_files(temp_files_glob):
    for temp_file in temp_files_glob:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"Removed related temp file: {temp_file}")
            except OSError as remove_err:
                print(f"Failed to remove temp file {temp_file}: {remove_err}")

def download_video(url, output_path):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'merge_output_format': 'mp4',
        'quiet': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return 1
    except yt_dlp.utils.DownloadError as e:
        path = output_path.split('\\')
        filename = path[-1].split('.')[0]
        delete_temp_files(glob.glob(os.path.join("\\".join(path[:-1]), f'{filename}.*')))
        return 0

def get_label_indices(csv_file):
    with open(csv_file, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        labels = {int(row[0]) for row in csvreader if row}
    return labels

def parse_and_download(input_path, output_path, target_labels, num_videos, max_length, min_height):
    # Iterate over Sports-1M JSON and download videos satisfying resolution, length and label crtieria
    # Stops at num_videos downloads

    num_downloaded = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(input_path, 'r') as file:
        data = json.load(file)
        split_path = input_path.split('\\')
        print(f'Processing Sports-1M dataset file {split_path[-1]}')
        
        for item in data:
            duration = item.get('duration', 0)
            height = item.get('height', 0)
            if duration > max_length or height < min_height: 
                # only download videos <=(max_length)s in length
                # partially doing this to save time and space, partially doing this because longer videos tend to be a lot less focused
                # (e.g. many minutes of no frames containing sports balls)
                continue

            labels = item.get('label487', [])
            labels_set = set(labels)

            if target_labels.intersection(labels_set):
                if num_downloaded >= num_videos:
                    break
                video_id = item.get('id')
                url = f'https://youtube.com/watch?v={video_id}'
                output_file = f'{num_downloaded}.mp4'
                print(f'Downloading {duration}s video at {url} (labels: {labels}) to {output_file}')
                dl_success = download_video(url, os.path.join(output_path, output_file))
                num_downloaded += dl_success

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-labels-csv", help="Path to CSV containing Sports-1M [label-idx],[label-name] pairs; restricts videos being downloaded", required=True)
    parser.add_argument("--sports-dataset-json", help="Path to Sports-1M JSON from which videos should be downloaded", required=True)
    parser.add_argument("--output-path", help="Path to directory to save videos to", required=True)
    return parser.parse_args()

def main():
    args = parse_cli_args()

    target_labels = get_label_indices(args.target_labels_csv)
    print(f'Target labels: {target_labels}')

    parse_and_download(args.sports_dataset_json, args.output_path, target_labels, num_videos=10, max_length=30, min_height=720)

if __name__ == '__main__':
    main()