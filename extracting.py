# extract_features.py

import os
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from dcnn import DCNN
from utils import encode_label_from_path, VehicleColorDataset
from sklearn.model_selection import train_test_split

# Dataset path and transform
path = "/home/datdq/1WorkSpace/Dataset_General/COLOR_TYPE/train"
image_list = glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True)
class_list = [encode_label_from_path(item) for item in image_list]

x_train, _, y_train, _ = train_test_split(image_list, class_list, train_size=0.5, stratify=class_list, shuffle=True, random_state=42)

transforms = Compose([
    Resize((227, 227)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = VehicleColorDataset(x_train, y_train, transforms)
train_loader = DataLoader(train_dataset, batch_size=115, shuffle=False, num_workers=4)

# Load trained model
model = DCNN(num_classes=8)
model.load_state_dict(torch.load("logs/DCNN_color/model_best.pt", map_location='cpu'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

conv5_features, fc6_features, fc7_features, labels_all = [], [], [], []

with torch.no_grad():
    for X, y in tqdm(train_loader, desc="Extracting features"):
        X = X.to(device)
        conv5, fc6, fc7 = model(X, return_features=True)
        conv5_features.append(conv5.cpu().numpy())
        fc6_features.append(fc6.cpu().numpy())
        fc7_features.append(fc7.cpu().numpy())
        labels_all.append(y.cpu().numpy())

# Save features
os.makedirs("features", exist_ok=True)
np.save("features/conv5.npy", np.vstack(conv5_features))
np.save("features/fc6.npy", np.vstack(fc6_features))
np.save("features/fc7.npy", np.vstack(fc7_features))
np.save("features/labels.npy", np.concatenate(labels_all))
print("Done extracting features and saving to disk.")