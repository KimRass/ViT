import pickle
from pathlib import Path

### CIFAR-100
DATA_DIR = "/Users/jongbeomkim/Documents/datasets/cifar-100-python/"
with open(Path(DATA_DIR)/"meta", mode="rb") as f:
    meta = pickle.load(f, encoding="bytes")
fine_label_names = meta[b"fine_label_names"]
CIFAR100_CLASSES = [i.decode("ascii") for i in fine_label_names]
N_CLASSES = 100
IMG_SIZE = 32

### ImageNet 1000
DATA_DIR = "/Users/jongbeomkim/Documents/datasets/imagenet-mini"

### Architecture
PATCH_SIZE = 16
N_LAYERS = 6
HIDDEN_DIM = 192
N_HEADS = 6

### Training
BATCH_SIZE = 4096 # "All models are trained with a batch size of 4096."
