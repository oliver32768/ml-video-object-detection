import os
import csv
import json
import glob
import yt_dlp

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

def main():
    target_labels = get_label_indices('relevant-labels.csv')
    print(f'Target labels: {target_labels}')

    train_input_path = os.path.join(os.getcwd(), 'sports-1m-dataset', 'sports1m_train.json')
    train_output_path = os.path.join(os.getcwd(), 'videos', 'train')

    val_input_path = os.path.join(os.getcwd(), 'sports-1m-dataset', 'sports1m_test.json')
    val_output_path = os.path.join(os.getcwd(), 'videos', 'val')

    parse_and_download(train_input_path, train_output_path, target_labels, num_videos=10, max_length=30, min_height=720)
    parse_and_download(val_input_path, val_output_path, target_labels, num_videos=2, max_length=30, min_height=720)

if __name__ == '__main__':
    main()