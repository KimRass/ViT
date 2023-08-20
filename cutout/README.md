# Paper Reading
- [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)
## Experiments
- Figure 3
    - <img src="https://user-images.githubusercontent.com/105417680/229458790-c47a4ef5-dce4-42de-b257-4e914188536e.png" width="700">
    - ***We find that model accuracy follows a parabolic trend, increasing proportionally to the Cutout size until an optimal point, after which accuracy again decreases and eventually drops below that of the baseline model.***
    - ***Based on these validation results we select a Cutout size of 16 × 16 pixels to use on CIFAR-10 and a Cutout size of 8 × 8 pixels for CIFAR-100 when training on the full datasets. Interestingly, it appears that as the number of classes increases, the optimal Cutout size decreases. This makes sense, as when more fine-grained detection is required then the context of the image will be less useful for identifying the category. Instead, smaller and more nuanced details are important.*** (Comment: Cutout size가 클수록 Cutout의 효과는 커집니다. Cutout의 효과는 모델이 이미지에서 dicriminative parts만이 아니라 non-discriminative parts를 포함한 전체 context를 보도록 하는 것입니다. number of classes가 커질수록 모델이 각 클래스를 구분하기 위해서는 이미지에서 discriminative parts를 보아야 하므로 작은 Cutout size가 필요한 것입니다.)
