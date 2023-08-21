# Paper Reading
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)

# Result
```python
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
- Test set에 대해 Top-5 accuracy 0.664 (240 epochs)
```python
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
- Test set에 대해 Top-5 accuracy 0.670 (240 epochs)
```python
N_LAYERS = 6
HIDDEN_DIM = 192
N_HEADS = 6
PATCH_SIZE = 4
HIDE_AND_SEEK = True
BATCH_SIZE = 2048
BASE_LR = 3e-3
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 0.3
```

# Research
## 23.08.20
- `F.gelu()` → `nn.Dropout()`의 순서가 되도록 Architecture를 변경했습니다. 순서를 반대로 할 경우 미분 값이 0이 되어 학습이 이루어지지 않는 현상이 발생함을 확인했습니다.
