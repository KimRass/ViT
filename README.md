# 'ViT' (Dosovitskiy et al., 2020) implementation from scratch in PyTorch
## Pre-trained Models
```python
DROP_PROB = 0.1
N_LAYERS = 6
HIDDEN_SIZE = 384
MLP_SIZE = 384
N_HEADS = 12
PATCH_SIZE = 4
BASE_LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 5e-5
WARMUP_EPOCHS = 5
SMOOTHING = 0.1
CUTMIX = False
CUTOUT = False
HIDE_AND_SEEK = False
BATCH_SIZE = 2048
N_EPOCHS = 300
```
### Trained on CIFAR-10 dataset for 300 epochs
- [vit_cifar10.pth](https://drive.google.com/file/d/1NkMB-WIDIwLIs-DvIxI39-K4TgQFq-nL/view?usp=sharing)
- Top-1 accuracy 0.864 on test set
```python
MEAN = (0.491, 0.482, 0.447)
STD = (0.248, 0.244, 0.261)
```
### Trained on CIFAR-100 dataset for 256 epochs
- [vit_cifar100.pth](https://drive.google.com/file/d/1vxH9c1q2BbHiFRN8JSlu3zj7ZBPvQYR8/view?usp=sharing)
- Top-1 accuracy 0.447 on test set
```python
MEAN = (0.507, 0.487, 0.441)
STD = (0.267, 0.256, 0.276)
```
## Implementation Details
- `F.gelu()` → `nn.Dropout()`의 순서가 되도록 Architecture를 변경했습니다. 순서를 반대로 할 경우 미분 값이 0이 되어 학습이 이루어지지 않는 현상이 발생함을 확인했습니다.
- CIFAR-100에 대해서 `N_LAYERS = 6, HIDDEN_SIZE = 384, N_HEADS = 6`일 때, `PATCH_SIZE = 16`일 때보다 `PATCH_SIZE = 8`일 때, 그리고 `PATCH_SIZE = 4`일 때 성능이 향상됐습니다.
- CIFAR-10과 CIFAR-100에 대해서 공통적으로 ViT-Base보다 작은 크기의 모델을 사용할 때 성능이 더 높았습니다.
## Studies
### Attention Map
- Original image
    - <img src="https://github.com/KimRass/ViT/assets/67457712/e2088a4c-8a5f-4193-ac72-2f4b2ede2928" width="500">
- head_fusion: "max", discard_ratio: 0.85
    - <img src="https://github.com/KimRass/ViT/assets/67457712/2b3f1ec6-aa2d-4980-b29c-3d90edaa1909" width="500">
### Position Embedding Similarity
- <img src="https://github.com/KimRass/ViT/assets/67457712/be0efc06-a4d8-4da7-8a11-ed6730da2994" width="500">
