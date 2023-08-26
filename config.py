import torch
import pickle
from pathlib import Path
# import sys

# sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/vit_from_scratch")

### Data
### CIFAR-10
DATA_DIR = "/home/user/cv/cifar-10-batches-py"
# DATA_DIR = "/Users/jongbeomkim/Documents/datasets/cifar-10-batches-py"
with open(Path(DATA_DIR)/"batches.meta", mode="rb") as f:
    meta = pickle.load(f, encoding="bytes")
fine_label_names = meta[b"label_names"]
CIFAR100_CLASSES = [i.decode("ascii") for i in fine_label_names]
N_CLASSES = len(CIFAR100_CLASSES)
IMG_SIZE = 32
MEAN = (0.491, 0.482, 0.447)
STD = (0.248, 0.244, 0.261)
### CIFAR-100
# DATA_DIR = "/home/user/cv/cifar-100-python/"
# with open(Path(DATA_DIR)/"meta", mode="rb") as f:
#     meta = pickle.load(f, encoding="bytes")
# fine_label_names = meta[b"fine_label_names"]
# CIFAR100_CLASSES = [i.decode("ascii") for i in fine_label_names]
# N_CLASSES = len(CIFAR100_CLASSES)
# IMG_SIZE = 32
# MEAN = (0.507, 0.487, 0.441)
# STD = (0.267, 0.256, 0.276)

### Architecture
DROP_PROB = 0.1
N_LAYERS = 6
HIDDEN_SIZE = 384
MLP_SIZE = 384
N_HEADS = 12
PATCH_SIZE = 4

### Optimizer
# "Adam with $beta_{1} = 0.9$, $beta_{2}= 0.999$, a batch size of 4096 and apply a high weight decay
# of 0.1, which we found to be useful for transfer of all models."
BASE_LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 5e-5
WARMUP_EPOCHS = 5

### Regularization
SMOOTHING = 0.1 # If `0`, do not employ label smoothing
CUTMIX = True
CUTOUT = False
HIDE_AND_SEEK = False

### Training
SEED = 17
# BATCH_SIZE = 4096 # "All models are trained with a batch size of 4096."
BATCH_SIZE = 2048
N_EPOCHS = 300
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
N_PRINT_EPOCHS = 4
N_VAL_EPOCHS = 4
CKPT_DIR = Path(__file__).parent/"checkpoints"

### Resume
CKPT_PATH = CKPT_DIR/"epoch_200_avg_acc_0.849.pth"
