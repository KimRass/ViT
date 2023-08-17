import torch
import pickle
from pathlib import Path

### CIFAR-100
# DATA_DIR = "/Users/jongbeomkim/Documents/datasets/cifar-100-python/"
DATA_DIR = "/home/user/cv/cifar-100-python/"
with open(Path(DATA_DIR)/"meta", mode="rb") as f:
    meta = pickle.load(f, encoding="bytes")
fine_label_names = meta[b"fine_label_names"]
CIFAR100_CLASSES = [i.decode("ascii") for i in fine_label_names]
N_CLASSES = 100
IMG_SIZE = 32

### ImageNet 1000
# DATA_DIR = "/Users/jongbeomkim/Documents/datasets/imagenet-mini"

### Architecture
N_LAYERS = 6
HIDDEN_DIM = 512
N_HEADS = 8
PATCH_SIZE = 16

### Training
SEED = 17
N_WORKERS = 0
N_GPUS = torch.cuda.device_count()
if N_GPUS > 0:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
MULTI_GPU = True
AUTOCAST = True
# BATCH_SIZE = 4096 # "All models are trained with a batch size of 4096."
BATCH_SIZE = 16 # "All models are trained with a batch size of 4096."
N_PRINT_STEPS = 100
N_CKPT_STEPS = 200
N_EPOCHS = 100