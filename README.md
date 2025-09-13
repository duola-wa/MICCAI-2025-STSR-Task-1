# MICCAI-2025-STSR-Task-1

## Environments and Requirements:
### 1. nnUNet Configuration
Install nnUNetv2 as below.  
You should meet the requirements of nnUNetv2, our method does not need any additional requirements.  
For more details, please refer to https://github.com/MIC-DKFZ/nnUNet  

```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
### 2. Dataset

Load Dataset from https://toothfairy3.grand-challenge.org/dataset/

### 3. Preprocessing

Conduct automatic preprocessing using nnUNet.

```
nnUNetv2_plan_and_preprocess -d 302 --verify_dataset_integrity
```


### 4. Training

Train by nnUNetv2. 

Run script:

```
nnUNetv2_train 302 3d_fullres
```


### 5. Inference

Test by nnUNetv2 with post-processing. 

Run script:

```
python test_3D.py
```
