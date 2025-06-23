import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from dcnn import DCNN
# Load the pre-trained model

transforms=Compose([Resize((227, 227)), 
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

MODEL_PATH = '/home/datdq/1WorkSpace/testing_out_paper/Spatial_Pyramid_DCNN_SVM_thingy/logs/DCNN_color/model_best.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DCNN(num_classes=8).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# Train SVM using the extracted features
X = np.load("features_fc7.npy")
# X = np.load("features_conv5.npy") # Just stick with fc7 for now cuz it is faster || and fc7 performs kinda the same as conv5
# print("Shape of X:", X.shape)
# X_tensor = torch.tensor(X)
# X_resized = F.interpolate(X_tensor, scale_factor=0.5, mode='bilinear', align_corners=False)
# X= X.reshape(X.shape[0], -1)
y = np.load("labels.npy")

clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

y_pred = clf.predict(X)
print("SVM training accuracy:", accuracy_score(y, y_pred))

img = transforms(Image.open("/home/datdq/1WorkSpace/Dataset_General/color/color_test/blue/snap_201202221603404020_晋JE0526_蓝_20120222160348_1.jpg").convert("RGB")).unsqueeze(0).to(device)

# Extract features
model.eval()
with torch.no_grad():
    conv5, fc6, fc7 = model(img, return_features=True)

# conv5_flattened = conv5.view(1, -1).cpu().numpy()
# Predict
y_pred = clf.predict(fc7.cpu().numpy())
# y_pred = clf.predict(conv5_flattened)
print("Predicted color class:", y_pred[0])