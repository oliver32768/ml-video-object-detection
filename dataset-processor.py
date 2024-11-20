from ultralytics import YOLO

def gen_pseudo_annotations(model, input_dir, output_dir):
    """
    model is pretrained/finetuned YOLOv10

    input_dir contains MP4 videos downloaded from YT

    output_dir should contain (*.jpg, *.txt) where JPEGs are video frames and TXTs are YOLO formatted AABB labels
    """

def main():
    pass

if __name__ == '__main__':
    main()