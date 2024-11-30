# Machine Learning Video Object Detection

Coding assignment with the objective of automatically generating pseudo-labels of sports ball detections in videos, and iteratively improving upon label quality

(Tested on Python 3.12.4)

## Setup

1. Clone repo:

```bash
git clone https://github.com/oliver32768/ml-video-object-detection.git
cd ml-video-object-detection
```

2. Download Sports-1M dataset JSON at `https://cs.stanford.edu/people/karpathy/deepvideo/sports1m_json.zip`, and extract to project directory

3. Create and activate virtual environment:

```bash
python -m venv .venv
./.venv/Scripts/activate
pip install -r requirements.txt
```

## Usage 

1. Download videos from Sports-1M dataset (example, see docstring in `dataset-downloader.py`):

```bash
python dataset-downloader.py --target-labels-csv relevant-labels.csv --sports-dataset-json sports-1m-dataset/sports1m_train.json --output-path videos/train
```

2. Generate first pass of pseudo-labels for downloaded videos (example, see docstring in `dataset-processor.py`):

```bash
python dataset-processor.py --video-dir videos/train --dataset-dir dataset/0
```

3. Finetune model on pseudo-labelled dataset (example, see docstring in `main.py`):

```bash
python main.py --dataset-dir dataset/0 --tag 0 --num-epochs 10
```

4. Generate second pass of pseudo-labels using finetuned model:

```bash
python dataset-processor.py --video-dir videos/train --dataset-dir dataset/1 --weights models/checkpoint_best-0.pth
```

5. Clean dataset from second pass and merge with dataset from first pass (example, see docstring in `dataset-cleaner.py`):

```bash
python dataset-cleaner.py --output-dataset-dir dataset/2 --input-dataset-dir dataset/1 --prev-dataset-dir dataset/0
```

6. (Optional) Repeat from Step 3, but:
* (Step 3) Use the output dataset from `dataset-cleaner` as the input dataset for `main`
* (Step 4) Use the output weights from `main` in `dataset-processor`
* (Step 5) Use the output dataset from `dataset-processor` as the input dataset for `dataset-cleaner`
* (Step 5) Use the output dataset from the previous run of `dataset-cleaner` as the prev-dataset for `dataset-cleaner`

Optionally, you can produce visualisations of any dataset produced by `dataset-processor` by running the following (example, see docstring in `dataset-visualisations.py`):

```bash
python dataset-visualisations.py --dataset-dir dataset/2
```

## Documentation

This project implements a scheme for generating pseudo-labels of sports-ball bounding boxes in video sequences, and then iteratively improving upon these pseudo-labels with no manual intervention whatsoever. It works in the following loop:

* Load pretrained FasterRCNN v1 model
* Generate pseudo-labels 
* Replace the FasterRCNN v1 head with a binary classifier (background vs. sports-ball) and finetune on these pseudo-labels
* Generate a new set of pseudo-labels
* Prune implausible labels using heuristics
* Merge any labels from the previous dataset that the new one produced false negatives for

### Design Considerations

**Model choice**: I chose FasterRCNN v1 as it requires less compute and significantly less training data than a SOTA Transformer based vision model such as Segment Anything (SAM), the latter consideration being crucial given time constraints. YOLO has similar resource and time constraints to FasterRCNN but produced qualitatively worse labels for this task (i.e. more false positives and negatives).

**Labelling Methodology**: I use a strict confidence threshold on bounding box labels to minimize false positives at the expense of more false negatives, and then train the model only on frames containing a nonzero amount of detections. The reasoning for this strems from two key considerations: 

1. I want to intentionally overfit the model to visual characteristics of the sports-balls in the dataset so that it can increase its confidence of ball presence for frames marked as false negatives in previous iterations. Over-fitting is bad in most contexts, but given that our task is to automate pseudo-labelling of a known dataset, it can be beneficial to sacrifice generalization for faster convergence.
2. The strict confidence threshold means each labelled video has very few to no false positives; any false positives are damaging to this loop as the model reinforces these errors and eventually collapses
3. Given this, I additionally implemented a step in each iteration that removes implausible bounding boxes using heuristics. This can be improved; see below.

**Improvements**:

1. **Dataset Cleaning**: Given more time, I would like to try computing deep features (e.g. using VGG19) of pixels inside bounding boxes and performing some form of outlier detection (e.g. isolation forests, or K-Means on dimensionality reduced (e.g. PCA, Autoencoders) features, etc.) on these in order to prune more false positives

2. **Temporal Considerations**: None of SAM, YOLO or FasterRCNN make explicit consideration of temporal aspects; they do not have knowledge of previous frames, and hence no knowledge of velocity. It may be possible to track correspondences between bounding box pixels and from this estimate state (position and velocity) which can be used in a Kalman Filter (or another temporal model) to assist labelling (e.g. by computing confidence as a function of FasterRCNN and KF confidences).
