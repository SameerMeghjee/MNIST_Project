🧠 MNIST Digit Recognition Project

This project implements a complete machine learning pipeline to classify handwritten digits using the MNIST dataset. It includes data preprocessing, model training, evaluation, error analysis, and deployment using a Streamlit web app.

---

## 📁 Project Structure

mnist-digit-classifier/
- ├── mnist_full_project.py # Full training & evaluation script
- ├── mnist_app.py # Streamlit app for digit prediction
- ├── rf_model.pkl # Trained Random Forest model
- ├── scaler.pkl # Saved StandardScaler object
- ├── README.md # Project documentation
- └── requirements.txt # Dependencies

---

## 📚 Dataset

- **Name**: MNIST (Modified National Institute of Standards and Technology)
- **Source**: `fetch_openml('mnist_784')`
- **Format**: 70,000 grayscale images, each 28x28 pixels (flattened into 784 features)

---

## 🚀 Models Used

| Model                | Accuracy |
|---------------------|----------|
| SGD Classifier       | ~89%     |
| Random Forest Classifier | >95%    |

- **SGDClassifier**: Used with hinge loss for linear SVM
- **RandomForestClassifier**: Robust ensemble-based model

---

## 📊 Performance Metrics

- **Accuracy**
- **Confusion Matrix**
- **Precision, Recall, F1 Score**
- **Error Analysis**: Identified common digit confusion patterns (e.g., 9 → 4, 5 → 3)

---

## 🔍 Error Analysis Highlights

| True Label | Predicted | Insight                              |
|------------|-----------|--------------------------------------|
| 9          | 4         | Top loop interpreted as tail         |
| 5          | 3         | Similar arcs and curves              |
| 7          | 1         | Both are tall and narrow             |

---

## 🌐 Streamlit Web App

The project includes a **Streamlit app** that allows users to upload a digit image and receive an instant prediction.

