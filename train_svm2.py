import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
# Load features
X = np.load("features/fc7.npy")  # Or fc6.npy, conv5.npy
y = np.load("features/labels.npy")

# Train and evaluate
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)
joblib.dump(clf, "svm_fc7.pkl")
y_pred = clf.predict(X)
print("SVM training accuracy:", accuracy_score(y, y_pred))