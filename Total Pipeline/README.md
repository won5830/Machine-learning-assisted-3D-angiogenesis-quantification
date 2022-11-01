## Installation
Before running, install pointnet2 library by running the following command:
```shell
cd utils/pointnet2
python setup.py install
cd ../..
```
## Data preparation
Download the angiogenesis dataset [here](https://kr.mathworks.com/matlabcentral/fileexchange/43400-skeleton3d). Prepared `h5` file contains N by 3 matrix of each chamber's original data and skeleton data extracted through *Deep Point Consolidation* module. 
```
Tot_data.h5
├── ...
├── chamber*
│   ├──skeleton
│   │  ├── Nx3 matrix
│   ├──original
│   │  ├── Mx3 matrix
├── ...
```
## Run 
Name of the dataset should be formatted as follows before running. Evaluation result will be saved in `checkpoint/exp_name` as `csv` file.  
* If you want to evaluate angiogenesis data through skeleton data that is already extracted with [deep point consolidation](https://kr.mathworks.com/matlabcentral/fileexchange/43400-skeleton3d), save skeleton data as `input_ply_data_skel*.ply` 
* If you want to evaluate angiogenesis data from scratch, save original data as `input_ply_data_origin*.ply`. Temporary implementation of knn-contraction will be used for skeleton extraction. **Not recommended** due to the absence of several optimization modules. Please use dpc module for meso-skeleton extraction. Variables can be further optimized at `knn-contraction` function in `skel_util.py`.  

```shell
## Evaluation through pre-extracted skeleton
python main.py --exp_name=221101_test --from_skel=True --model=sphadgcnn --model_path=pretrained/model.t7 --batch_size=2 --num_points=2400

## Evaluation from scratch
python main.py --exp_name=221101_test --from_skel=False --model=sphadgcnn --model_path=pretrained/model.t7 --batch_size=2 --num_points=2400
```
