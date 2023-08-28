# Result
## CIFAR-10
```python
MEAN = (0.491, 0.482, 0.447)
STD = (0.248, 0.244, 0.261)
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
- Top-1 accuracy 0.864 on test set at 300 epoch
## CIFAR-100
```python
MEAN = (0.507, 0.487, 0.441)
STD = (0.267, 0.256, 0.276)
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
- Top-1 accuracy 0.447 on test set at 256 epoch

# Research
- `F.gelu()` → `nn.Dropout()`의 순서가 되도록 Architecture를 변경했습니다. 순서를 반대로 할 경우 미분 값이 0이 되어 학습이 이루어지지 않는 현상이 발생함을 확인했습니다.
- CIFAR-100에 대해서 `N_LAYERS = 6, HIDDEN_SIZE = 384, N_HEADS = 6`일 때, `PATCH_SIZE = 16`일 때보다 `PATCH_SIZE = 8`일 때, 그리고 `PATCH_SIZE = 4`일 때 성능이 향상됐습니다.
- CIFAR-10과 CIFAR-100에 대해서 공통적으로 ViT-Base보다 작은 크기의 모델을 사용할 때 성능이 향상됐습니다.

# References:
- [1] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [2] [Understanding Why ViT Trains Badly on Small Datasets: An Intuitive Perspective](https://arxiv.org/abs/2302.03751)