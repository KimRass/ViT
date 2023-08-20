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
N_CLASSES = len(CIFAR100_CLASSES)
IMG_SIZE = 32

### ImageNet 1000
# DATA_DIR = "/Users/jongbeomkim/Documents/datasets/imagenet-mini"
# IMG_SIZE = 224 # "All training is done on resolution 224."

### Architecture
DROP_PROB = 0.1
N_LAYERS = 6
HIDDEN_DIM = 384
N_HEADS = 6
PATCH_SIZE = 16
# N_LAYERS = 12
# HIDDEN_DIM = 768
# N_HEADS = 12
# PATCH_SIZE = 16

### Optimizer
# "Adam with $beta_{1} = 0.9$, $beta_{2}= 0.999$, a batch size of 4096 and apply a high weight decay
# of 0.1, which we found to be useful for transfer of all models."
# LR = 0.01
BASE_LR = 3e-3
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.3
# WEIGHT_DECAY = 0.1

### Training
SEED = 17
N_WORKERS = 6
N_GPUS = torch.cuda.device_count()
if N_GPUS > 0:
    DEVICE = torch.device("cuda")
    print(f"""Using {N_GPUS} GPU(s).""")
else:
    DEVICE = torch.device("cpu")
    print(f"""Using CPU(s).""")
MULTI_GPU = True
AUTOCAST = True
# BATCH_SIZE = 4096 # "All models are trained with a batch size of 4096."
BATCH_SIZE = 8192 # For CIFAR-100
N_PRINT_EPOCHS = 2
N_VAL_EPOCHS = 4
N_CKPT_EPOCHS = 30
N_EPOCHS = 500
CKPT_DIR = Path(__file__).parent/"checkpoints"

### Resume
CKPT_PATH = CKPT_DIR/"300_6.pth"