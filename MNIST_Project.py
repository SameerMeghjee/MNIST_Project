import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(np.uint8)

# Split into training and testing sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

# Normalize for SGD
print("Scaling input...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.transform(X_test.astype(np.float64))

# Train SGD Classifier
print("Training SGD Classifier...")
sgd_clf = SGDClassifier(loss='hinge', random_state=42)
sgd_clf.fit(X_train_scaled, y_train)
y_pred_sgd = sgd_clf.predict(X_test_scaled)

# Train Random Forest Classifier
print("Training Random Forest Classifier...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# Evaluate both classifiers
print("\n--- SGD Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred_sgd))
print(classification_report(y_test, y_pred_sgd))

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion matrix for RF
conf_mx = confusion_matrix(y_test, y_pred_rf)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Save models
print("Saving models...")
joblib.dump(sgd_clf, "sgd_model.pkl")
joblib.dump(rf_clf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Models saved.")
# Load and test saved models
print("Loading saved models...")
sgd_clf_loaded = joblib.load("sgd_model.pkl")
rf_clf_loaded = joblib.load("rf_model.pkl")
scaler_loaded = joblib.load("scaler.pkl")

# Test loaded models
print("Testing loaded models...")
X_test_scaled_loaded = scaler_loaded.transform(X_test.astype(np.float64))
y_pred_sgd_loaded = sgd_clf_loaded.predict(X_test_scaled_loaded)
y_pred_rf_loaded = rf_clf_loaded.predict(X_test)

# Evaluate loaded models
print("\n--- Loaded SGD Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred_sgd_loaded))
print(classification_report(y_test, y_pred_sgd_loaded))

print("\n--- Loaded Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_loaded))
print(classification_report(y_test, y_pred_rf_loaded))

# Confusion matrix for loaded RF
conf_mx_loaded = confusion_matrix(y_test, y_pred_rf_loaded)
plt.matshow(conf_mx_loaded, cmap=plt.cm.gray)
plt.title("Confusion Matrix - Loaded Random Forest")
plt.show()