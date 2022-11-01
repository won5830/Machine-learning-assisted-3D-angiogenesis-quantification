## Data Preparation 
Download the data for skeleton segmentation [here](https://kr.mathworks.com/matlabcentral/fileexchange/43400-skeleton3d) and save in `data/test_skel_ply_hdf5_data_train*.h5`

## Run 
You can run different modes with following codes. Trained model can later be used at 3D evaluation pipeline.
```shell
## K-Fold training for model evaluation
python main.py --kfold=True --fold_num=5  --exp_name=221101_test --model=sphadgcnn --use_sgd=True --num_points=2400 --k=30 --dropout=0.3 --emb_dims=1024 --batch_size=2 --test_batch_size=2 --epochs=1500

## Train without K-Fold
python main.py --kfold=False --exp_name=221101_test --model=sphadgcnn --use_sgd=True --num_points=2400 --k=30 --dropout=0.3 --emb_dims=1024 --batch_size=2 --test_batch_size=2 --epochs=1500

## Evaluation on test set
python main.py --eval=True --kfold=False --exp_name=221101_test --model=sphadgcnn --use_sgd=True --num_points=2400 --k=30 --dropout=0.3 --emb_dims=1024 --batch_size=2 --test_batch_size=2 --epochs=1500
```
