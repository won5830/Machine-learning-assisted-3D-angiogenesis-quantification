# Machine Learning Assisted 3D Angiogenesis Quantification
This repository contains the pytorch code for the paper: Machine learning-aided quantification of 3D angiogenic vasculature in multiculture microfluidic platform

## Installation
The latest codes are tested on Ubuntu 16.04, CUDA10.1, PyTorch 1.6 and Python 3.7:
```
conda install pytorch torchvision pytorch-cuda=11.3 -c pytorch -c nvidia
```

## Model Training
### Data Preparation 
Download the data for skeleton segmentation [here](https://kr.mathworks.com/matlabcentral/fileexchange/43400-skeleton3d) and save in `data/test_skel_ply_hdf5_data_train*.h5`

