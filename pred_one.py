# predict_one.py

from PIL import Image
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from dcnn import DCNN
from sklearn.svm import SVC
import joblib
import time
# Load model and weights
total_start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DCNN(num_classes=8).to(device)
model.load_state_dict(torch.load("logs/DCNN_color/model_best.pt", map_location=device))
model.eval()

# Load SVM
clf = joblib.load("/home/datdq/1WorkSpace/testing_out_paper/Spatial_Pyramid_DCNN_SVM_thingy/svm_fc7.pkl")

# Transform
transform = Compose([
    Resize((227, 227)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_path = "/home/datdq/1WorkSpace/Dataset_General/color/color_test/blue/snap_201203021222509970_蒙B47859_黄_20120302122301_2.jpg"
img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

# Extract fc7 feature
with torch.no_grad():
    _, _, fc7 = model(img, return_features=True)
st = time.time()
pred = clf.predict(fc7.cpu().numpy())
et = time.time()

print("Time taken for prediction:", et - st, "seconds")
print("Total time taken:", et - total_start, "seconds")
print("Predicted class:", pred[0])
