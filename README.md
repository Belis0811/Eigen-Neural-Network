# Eigen Neural Network: Unlocking Generalizable Vision with Eigenbasis
This repository introduces Eigen Neural Network (ENN), a learning framework built to overcome the memory overhead, update-locking latency, and biological implausibility imposed by global back-propagation in modern vision and multimodal systems.

By re-parameterizing every layer in a shared, orthonormal eigenbasis, ENN presents a new architectural paradigm that directly remedies the representational deficiencies of BP, leading to enhanced performance and enabling a more efficient, parallelizable training regime. This repo is built upon our paper
>[Eigen Neural Network: Unlocking Generalizable Vision with Eigenbasis]
>


## Overview
Paper is implemented with official pytorch
![Overview Image](figures/overview.png?raw=true "Overview of the proposed ENN")

This high-level overview figure clearly illustrates the detailed architecture of the proposed ENN architecture.

## Requirements
* **Python** 3.8+
* **PyTorch** 2.0.0+
* **torchvision** 0.15.0+
*  **NumPy**,**scikit-image**,**matplotlib**,etc
  
Please go to the `requirement.txt` file to check all dependencies.

Or run the following code to install:
```
pip install -r requirements.txt
```
## Data Downloading
Put the Tiny ImageNet dataset into the root folder, then name the dataset folder `tiny-imagenet-200`. The dataset could be found at https://www.kaggle.com/c/tiny-imagenet/data
Put the ImageNet dataset into the root folder, then name the dataset folder `imagenet`. Note that Imagenet needs some more preprocessing. Please refer to https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh for details. The ImageNet dataset is located at https://image-net.org/; you could download it by yourself.


## Training
*Note: The hyperparameters of the codes may not reflect the latest configuration. Please refer to the Final Notes and consult with AC or PB for the most current hyperparameter settings.*

To train the ENN model and obtain evaluation metrics for image classification or image retrieval, execute the relevant Python script as follows:
```
python cifar100_r101_bp.py
```

## Results
![alt text](figures/comparison.png?raw=true "Comparison with SOTA Models")

ENN is outperforming most of the state-of-the-art models for image classification. The best results are highlighted in **bold**, and the second best are **underlined**.

![alt text](figures/performance.png?raw=true "Error rate of different methods")

ENN also showed superiority on image retrieval tasks.
