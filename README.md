### Dataset
[VeRI Dataset](https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset)

### How to run
Run 
- training.ipynb
- take the model_best.pt in logs/DCNN_color and put it in extracting.py
- take the fc7.npy (or fc6 or conv5) and putin train_svm2.py
- get the .pkl file and use in pred_multi for multi image prediction, pred_one.py for predict one image. (surely the pred_multi can do the same as pred_one thou.)

### Credit
Link to paper: \
[Vehicle Color Recognition With Spatial Pyramid Deep Learning](https://ieeexplore.ieee.org/document/7118723) Chuanping Hu; Xiang Bai; Li Qi; Pan Chen; Gengjian Xue; Lin Mei
