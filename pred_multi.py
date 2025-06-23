from PIL import Image
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from dcnn import DCNN
from sklearn.svm import SVC
import joblib
import glob
import time
import os
# Load model and weights
total_start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DCNN(num_classes=8).to(device)
model.load_state_dict(torch.load("logs/DCNN_color/model_best.pt", map_location=device))
model.eval()

clf = joblib.load("/home/datdq/1WorkSpace/testing_out_paper/thingcopy/svm_weight/svm_fc7.pkl")
transform = Compose([
    Resize((227, 227)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_folder = "/home/datdq/1WorkSpace/Dataset_General/color/color_test/"
labels = ['black', 'blue', 'gray', 'green', 'orange', 'red', 'white', 'yellow']
correct = 0
total = 0


# with torch.no_grad():
#     for label in labels:
#         img_paths = glob.glob(os.path.join(base_folder, label, '*.jpg'))
#         for img_path in img_paths:
#             try:
#                 img = Image.open(img_path).convert("RGB")
#                 img_tensor = transform(img).unsqueeze(0).to(device)

#                 # Feature extraction
#                 _, _, fc7 = model(img_tensor, return_features=True)
#                 pred = clf.predict(fc7.cpu().numpy())[0]
#                 pred_label = labels[pred]

#                 result = f"{label}/{os.path.basename(img_path)} || GT: {label} || Pred: {pred_label}"
#                 print(result)

#                 total += 1
#                 if pred_label == label:
#                     correct += 1
#             except Exception as e:
#                 print(f"Error with image {img_path}: {e}")

# acc = 100 * correct / total if total > 0 else 0
# print(f"\nTotal Accuracy: {acc:.2f}% on {total} images")
# print(f"Total Time: {time.time() - total_start:.2f} seconds")

labels = ['black', 'blue', 'gray', 'green', 'orange', 'red', 'white', 'yellow']

# Base folder containing test images
base_folder = "/home/datdq/1WorkSpace/Dataset_General/color/color_test/"  # Replace with your folder path

# Mapping of folder names to trained labels
folder_to_label = {
    'black': 'black',
    'blue': 'blue',
    'blue2': 'blue',  # Map blue2 (cyan) to blue
    'gray': 'gray',
    'green': 'green',
    'red': 'red',
    'white': 'white',
    'yellow': 'yellow'  # Yellow folder contains both yellow and orange images
}

# Function to remap predictions (orange -> yellow)
def remap_prediction(pred_label):
    return 'yellow' if pred_label == 'orange' else pred_label

total = 0
correct = 0
total_start = time.time()

with torch.no_grad():
    # Iterate over actual folders in base_folder
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip non-directory files
        
        # Map folder name to trained label
        if folder_name not in folder_to_label:
            print(f"Warning: Folder {folder_name} not mapped to any trained label. Skipping.")
            continue
        ground_truth_label = folder_to_label[folder_name]
        
        # Ensure the ground truth label is in the trained labels
        if ground_truth_label not in labels:
            print(f"Error: Mapped label {ground_truth_label} for folder {folder_name} not in trained labels. Skipping.")
            continue
        
        # Get all .jpg images in the folder
        img_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        
        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)

                # Feature extraction
                _, _, fc7 = model(img_tensor, return_features=True)
                pred = clf.predict(fc7.cpu().numpy())[0]
                pred_label = labels[pred]
                
                # Remap prediction: orange -> yellow
                remapped_pred_label = remap_prediction(pred_label)

                result = f"{folder_name}/{os.path.basename(img_path)} || GT: {ground_truth_label} || Pred: {remapped_pred_label}"
                print(result)

                total += 1
                if remapped_pred_label == ground_truth_label:
                    correct += 1
            except Exception as e:
                print(f"Error with image {img_path}: {e}")

# Calculate and print accuracy
acc = 100 * correct / total if total > 0 else 0
print(f"\nTotal Accuracy: {acc:.2f}% on {total} images")
print(f"Total Time: {time.time() - total_start:.2f} seconds")