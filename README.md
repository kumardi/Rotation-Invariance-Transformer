# Rotation-Invariance-Transformer
The repository contains the implementation of RiT and its application with ResNet18 as deescribed in the following two papers. Also the following references must be cited if using this code:

1. Kumar, D., & Sharma, D. (2023). Feature map augmentation to improve scale invariance in convolutional neural networks. Journal of Artificial Intelligence and Soft Computing Research, 13(1), 51-74.
2. Kumar, D., Sharma, D., & Goecke, R. (2020, February). Feature map augmentation to improve rotation invariance in convolutional neural networks. In International Conference on Advanced Concepts for Intelligent Vision Systems (pp. 348-359). Cham: Springer International Publishing.

**Description of contents:**
Functions.py - contains the implementation of the RiT module as described in the above papers.
Model_RiT.py - contains the implementation of the RiT with ResNet18 in the basic configuration (where RiT layer is placed at end of the convolutional pipeline)
Dataset.py - contains the implementation of the dataloader functions for Imagehoof dataset. Further details, download links for the dataset are available from the file.
main.py - has the basic train/test code for ResNet18+RiT 

Requirements:
1. Imagehoof dataset - https://github.com/fastai/imagenette
2. pytorch (min version) 3.6.8
3. Cuda/GPU support - preferable


