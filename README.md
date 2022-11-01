# Machine Learning Assisted 3D Angiogenesis Quantification
This repository contains the pytorch code for the paper: Machine learning-aided quantification of 3D angiogenic vasculature in multiculture microfluidic platform

## Installation
The latest codes are tested on Ubuntu 16.04, CUDA10.1, PyTorch 1.6 and Python 3.7:
```shell
conda install pytorch torchvision pytorch-cuda=11.3 -c pytorch -c nvidia
```

## Model Training
### Data Preparation 
Download the data for skeleton segmentation [here](https://kr.mathworks.com/matlabcentral/fileexchange/43400-skeleton3d) and save in `data/test_skel_ply_hdf5_data_train*.h5`

### Run 
You can run different modes with following codes.
```shell
## K-Fold training for model evaluation
python main.py --kfold=True --fold_num=5  --exp_name=221101_test --model=sphadgcnn --use_sgd=True --num_points=2400 --k=30 --dropout=0.3 --emb_dims=1024 --batch_size=2 --test_batch_size=2 --epochs=1500

## Train without K-Fold
python main.py --kfold=False --exp_name=221101_test --model=sphadgcnn --use_sgd=True --num_points=2400 --k=30 --dropout=0.3 --emb_dims=1024 --batch_size=2 --test_batch_size=2 --epochs=1500

## Evaluation on test set
python main.py --eval=True --kfold=False --exp_name=221101_test --model=sphadgcnn --use_sgd=True --num_points=2400 --k=30 --dropout=0.3 --emb_dims=1024 --batch_size=2 --test_batch_size=2 --epochs=1500
```
