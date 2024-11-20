import os
import csv
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
        labels = {row[0] for row in csvreader if row}
    return labels

def parse_and_download(input_path, output_path, target_labels, num_videos):
    num_downloaded = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(input_path, 'r') as file:
        for line in file:
            url, _, labels_str = line.partition(' ')
            labels = labels_str.strip().split(',')
            
            if target_labels.intersection(labels):
                if num_downloaded >= num_videos:
                    break
                output_file = str(num_downloaded) + '.mp4'
                print(f'Downloading {url} (labels: {labels}) to {output_file}')
                dl_success = download_video(url, os.path.join(output_path, output_file))
                num_downloaded += dl_success

def main():
    target_labels = get_label_indices('relevant-labels.csv')
    print(f'Target labels: {target_labels}')

    train_input_path = os.path.join(os.getcwd(), 'sports-1m-dataset', 'original', 'train_partition.txt')
    train_output_path = os.path.join(os.getcwd(), 'dataset', 'train', 'videos')

    test_input_path = os.path.join(os.getcwd(), 'sports-1m-dataset', 'original', 'test_partition.txt')
    test_output_path = os.path.join(os.getcwd(), 'dataset', 'test', 'videos')

    parse_and_download(train_input_path, train_output_path, target_labels, num_videos=10)
    parse_and_download(test_input_path, test_output_path, target_labels, num_videos=2)

if __name__ == '__main__':
    main()