# Lab 01 — Results Tables

## Table 1: Transfer-Learning Model Accuracy Comparison

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|
| AlexNet | 70.31 | 75.03 | 70.31 | 68.47 | 89.32 |
| VGG16 | 62.50 | 52.39 | 62.50 | 54.99 | 88.22 |
| VGG19 | 75.00 | 85.30 | 75.00 | 68.01 | 96.88 |
| ResNet18 | 67.19 | 77.65 | 67.19 | 62.31 | 92.38 |
| ResNet50 | 68.75 | 73.76 | 68.75 | 66.48 | 87.63 |
| ResNet101 | 64.06 | 62.08 | 64.06 | 59.18 | 93.20 |
| DenseNet121 | 68.75 | 69.34 | 68.75 | 64.42 | 88.93 |
| EfficientNet-B0 | 70.31 | 78.75 | 70.31 | 62.90 | 94.37 |

## Training Summary (Best-CNN Selection)

| Model | Best Epoch | Best Validation Accuracy (%) | Final Test Accuracy (%) |
|---|---|---|---|
| DenseNet121 | 9 | 83.79 | 68.75 |
| EfficientNet-B0 | 6 | 83.49 | 70.31 |
| ResNet50 | 10 | 82.87 | 68.75 |
| VGG19 | 8 | 82.57 | 75.00 |
| ResNet101 | 6 | 81.65 | 64.06 |
| VGG16 | 8 | 81.35 | 62.50 |
| ResNet18 | 5 | 81.35 | 67.19 |
| AlexNet | 10 | 79.82 | 70.31 |

## Table 2: Deep Features (DenseNet121) + Classical Classifiers

| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|---|
| Deep Features | Logistic Regression | 71.88 | 76.15 | 71.88 | 69.02 | 89.84 |
| Deep Features | Decision Tree | 68.75 | 73.33 | 68.75 | 65.88 | 79.17 |
| Deep Features | Random Forest | 70.31 | 68.89 | 70.31 | 65.47 | 90.61 |
| Deep Features | K-Nearest Neighbors (KNN) | 71.88 | 74.36 | 71.88 | 69.41 | 85.53 |
| Deep Features | Linear SVM | 70.31 | 73.47 | 70.31 | 66.36 | 89.84 |
| Deep Features | RBF-SVM | 68.75 | 71.86 | 68.75 | 63.99 | 90.85 |
| Deep Features | XGBoost | 68.75 | 70.08 | 68.75 | 63.69 | 87.60 |

## Table 3: Computational Efficiency

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
|---|---|---|---|---|---|
| AlexNet | 57.02 | 217.52 | 1.42 | 2.24 | 70.31 |
| VGG16 | 134.28 | 512.24 | 30.93 | 9.13 | 62.50 |
| VGG19 | 139.59 | 532.49 | 39.26 | 10.97 | 75.00 |
| ResNet18 | 11.18 | 42.72 | 3.65 | 2.81 | 67.19 |
| ResNet50 | 23.52 | 90.01 | 8.26 | 6.52 | 68.75 |
| DenseNet121 | 6.96 | 27.13 | 5.79 | 15.31 | 68.75 |
| EfficientNet-B0 | 4.01 | 15.60 | 0.83 | 20.57 | 70.31 |
