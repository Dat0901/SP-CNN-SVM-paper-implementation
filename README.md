### Paper idea
![image](https://github.com/user-attachments/assets/41d91525-0a0e-4d0d-95ed-283ae3598d4e)


### Dataset
[VeRI Dataset](https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset)

### How to run
Run 
- run all cell in `training.ipynb`
- take the `model_best.pt` in logs/DCNN_color and put it in `extracting.py`
- take the `fc7.npy` (or fc6 or conv5) and put in `train_svm2.py` (if use conv5 then need a little tweak to make 4D->2D to fit SVC)
- get the `svm_fc7.pkl` file and use in `pred_multi.py` for multi image prediction, `pred_one.py` for predict one image. (surely the pred_multi can do the same as pred_one thou.)(Remember to change test set)

### Credit
Link to paper: \
[Vehicle Color Recognition With Spatial Pyramid Deep Learning](https://ieeexplore.ieee.org/document/7118723) Chuanping Hu; Xiang Bai; Li Qi; Pan Chen; Gengjian Xue; Lin Mei

### TODO
- [ ] Add Spatial Pyramid 2x2
- [ ] Be able to use GPU
