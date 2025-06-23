import os
import glob
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from dcnn import DCNN
from utils import encode_label_from_path, VehicleColorDataset

path = "/home/datdq/1WorkSpace/Dataset_General/COLOR_TYPE/train"
image_list = glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True)
class_list = [encode_label_from_path(item) for item in image_list]
print("Total Images: ", len(image_list))
print("Total Classes: ", len(set(class_list)))
# Splitting the Dataset
x_train, x_test , y_train , y_test = train_test_split(image_list, class_list, train_size= 0.5 , stratify=class_list , shuffle=True, random_state=42)
transforms=Compose([Resize((227, 227)), 
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# Create DataLoader for training and testing datasets
train_dataset = VehicleColorDataset( x_train , y_train , transforms)
train_data_loader = DataLoader(train_dataset,batch_size=115, shuffle=True, num_workers=4)
test_dataset = VehicleColorDataset(x_test, y_test,transforms)
test_data_loader = DataLoader(test_dataset, batch_size=115, num_workers=4)

MODEL_PATH='/home/datdq/1WorkSpace/testing_out_paper/Spatial_Pyramid_DCNN_SVM_thingy/logs/DCNN_color/model_best.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DCNN(num_classes=8).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()



conv5_features, fc6_features, fc7_features, labels_all = [], [], [], []
with torch.no_grad():
    for X, y in tqdm(train_data_loader, desc="Extracting Features"):
        X = X.to(device)
        conv5, fc6, fc7 = model(X, return_features=True)
        conv5_features.append(conv5.cpu())
        fc6_features.append(fc6.cpu())
        fc7_features.append(fc7.cpu())
        labels_all.append(y.cpu())

np.save("features_conv5.npy", np.vstack(conv5_features))
np.save("features_fc6.npy", np.vstack(fc6_features))
np.save("features_fc7.npy", np.vstack(fc7_features))
np.save("labels.npy", np.concatenate(labels_all))