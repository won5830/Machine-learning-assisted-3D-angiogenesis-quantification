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
You can run different modes with following codes. Trained model can later be used at 3D evaluation pipeline.
```shell
## K-Fold training for model evaluation
python main.py --kfold=True --fold_num=5  --exp_name=221101_test --model=sphadgcnn --use_sgd=True --num_points=2400 --k=30 --dropout=0.3 --emb_dims=1024 --batch_size=2 --test_batch_size=2 --epochs=1500

## Train without K-Fold
python main.py --kfold=False --exp_name=221101_test --model=sphadgcnn --use_sgd=True --num_points=2400 --k=30 --dropout=0.3 --emb_dims=1024 --batch_size=2 --test_batch_size=2 --epochs=1500

## Evaluation on test set
python main.py --eval=True --kfold=False --exp_name=221101_test --model=sphadgcnn --use_sgd=True --num_points=2400 --k=30 --dropout=0.3 --emb_dims=1024 --batch_size=2 --test_batch_size=2 --epochs=1500
```

## Total Pipeline

### Installation
Before running, install pointnet2 library by running the following command:
```shell
cd utils/pointnet2
python setup.py install
cd ../..
```
### Data preparation
Download the angiogenesis dataset [here](https://kr.mathworks.com/matlabcentral/fileexchange/43400-skeleton3d).
Pointnet2.PyTorch
├── pointnet2
├── tools
│   ├──data
│   │  ├── KITTI
│   │  │   ├── ImageSets
│   │  │   ├── object
│   │  │   │   ├──training
│   │  │   │      ├──calib & velodyne & label_2 & image_2
### Run 
Name of the dataset should be formatted as follows before running. Evaluation result will be saved in `checkpoint/exp_name` as `csv` file.  
* If you want to evaluate angiogenesis data through skeleton data that is already extracted with [deep point consolidation](https://kr.mathworks.com/matlabcentral/fileexchange/43400-skeleton3d), save skeleton data as `input_ply_data_skel*.ply` 
* If you want to evaluate angiogenesis data from scratch, save original data as `input_ply_data_origin*.ply`. Temporary implementation of knn-contraction will be used for skeleton extraction. **Not recommended** due to the absence of several optimization modules. Variables can be further optimized at `knn-contraction` function in `skel_util.py`.  

```shell
## Evaluation through pre-extracted skeleton
python main.py --exp_name=221101_test --from_skel=True --model=sphadgcnn --model_path=pretrained/model.t7 --batch_size=2 --num_points=2400

## Evaluation from scratch
python main.py --exp_name=221101_test --from_skel=False --model=sphadgcnn --model_path=pretrained/model.t7 --batch_size=2 --num_points=2400
```
