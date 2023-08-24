# Result
## CIFAR-100
- 해상도 32×32에 대해서
```python
N_LAYERS = 6
HIDDEN_DIM = 384
N_HEADS = 6
```
- 일 때, `PATCH_SIZE = 16`일 때보다 `PATCH_SIZE = 8`일 때, 그리고 `PATCH_SIZE = 4`일 때 성능이 향상됐습니다. (Test set에 대한 Top-5 accuracy가 각각 0.547, 0.664, 0.670, 그리고 각각 420 epochs, 240 epochs, 240 epochs에서 수렴)
- Hide-and-Seek를 적용하면 학습이 더 오래 걸릴뿐만 아니라 성능도 오히려 하락하는데, 이는 이미지의 해상도가 매우 낮고 (32×32) CIFAR-100 데이터셋의 크기가 작기 때문이 아닐까 싶습니다.
- 반대로 CutMix를 적용하면 학습이 더 빨라짐을 확인했습니다.
- Label smoothing으로 사용할 때 모델의 학습 속도가 빨라짐을 확인했습니다.
- Cosine learning rate schedule (with warm-up)을 사용할 때 학습이 빨라짐을 확인했습니다.
```python
N_LAYERS = 6
HIDDEN_DIM = 192
N_HEADS = 6
PATCH_SIZE = 4
HIDE_AND_SEEK = False
BATCH_SIZE = 2048
BASE_LR = 3e-3
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.3
```
- Test set에 대해 Top-1 accuracy 0.527 (132 epochs)
## CIFAR-10
```python
DROP_PROB = 0.1
N_LAYERS = 6
HIDDEN_DIM = 192
N_HEADS = 6
PATCH_SIZE = 4
SMOOTHING = 0.1
CUTMIX = False
HIDE_AND_SEEK = False
BATCH_SIZE = 2048
BASE_LR = 3e-3
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.3
```
- Test set에 대해 Top-1 accuracy 0.789 (296 epochs)
<!-- ```python
N_LAYERS = 6
HIDDEN_DIM = 384
N_HEADS = 6
PATCH_SIZE = 16
HIDE_AND_SEEK = True
BATCH_SIZE = 8192
BASE_LR = 3e-3
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.3
```
- Test set에 대해 Top-5 accuracy 0.547 (420 epochs)
```python
N_LAYERS = 6
HIDDEN_DIM = 384
N_HEADS = 6
PATCH_SIZE = 8
HIDE_AND_SEEK = True
BATCH_SIZE = 8192
BASE_LR = 3e-3
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.3
```
- Test set에 대해 Top-5 accuracy 0.664 (240 epochs) -->
<!-- ```python
N_LAYERS = 6
HIDDEN_DIM = 384
N_HEADS = 6
PATCH_SIZE = 4
HIDE_AND_SEEK = True
BATCH_SIZE = 2048
BASE_LR = 3e-3
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.3
```
- Test set에 대해 Top-5 accuracy 0.670 (240 epochs) -->

# Research
## 23.08.20
- `F.gelu()` → `nn.Dropout()`의 순서가 되도록 Architecture를 변경했습니다. 순서를 반대로 할 경우 미분 값이 0이 되어 학습이 이루어지지 않는 현상이 발생함을 확인했습니다.

# References:
- [1] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [2] [Understanding Why ViT Trains Badly on Small Datasets: An Intuitive Perspective](https://arxiv.org/abs/2302.03751)