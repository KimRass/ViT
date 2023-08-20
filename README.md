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

# Research
## 23.08.17
- 
```
/home/user/cv/lib/python3.8/site-packages/torch/autograd/__init__.py:200: UserWarning: Error detected in AddmmBackward0. Traceback of forward call that caused the error:
  File "/usr/lib/python3.8/threading.py", line 890, in _bootstrap
    self._bootstrap_inner()
  File "/usr/lib/python3.8/threading.py", line 932, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.8/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
  File "/home/user/cv/lib/python3.8/site-packages/torch/nn/parallel/parallel_apply.py", line 64, in _worker
    output = module(*input, **kwargs)
  File "/home/user/cv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/user/cv/vit_from_scratch/model.py", line 178, in forward
    x = self.tf_enc(x)
  File "/home/user/cv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/user/cv/vit_from_scratch/model.py", line 131, in forward
    x = enc_layer(x)
  File "/home/user/cv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/user/cv/vit_from_scratch/model.py", line 110, in forward
    ff_output = self.ff(x)
  File "/home/user/cv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/user/cv/vit_from_scratch/model.py", line 83, in forward
    x = self.w1(x)
  File "/home/user/cv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/user/cv/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
 (Triggered internally at ../torch/csrc/autograd/python_anomaly_mode.cpp:114.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
Traceback (most recent call last):
  File "train.py", line 125, in <module>
    loss.backward()
  File "/home/user/cv/lib/python3.8/site-packages/torch/_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "/home/user/cv/lib/python3.8/site-packages/torch/autograd/__init__.py", line 200, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [34816, 512]], which is output 0 of AsStridedBackward0, is at version 1; expected version 0 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
```
