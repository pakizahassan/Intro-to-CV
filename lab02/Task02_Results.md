# Task 02 — Image Filtering: Results Tables

## VGG19 — All Filters

| Model | Filter | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Macro-F1 (%) | Balanced Accuracy (%) | AUC (%) |
|---|---|---|---|---|---|---|---|---|
| VGG19 | Original | 70.31 | 72.26 | 70.31 | 67.99 | 67.99 | 70.31 | 94.30 |
| VGG19 | Mean | 79.69 | 84.88 | 79.69 | 77.53 | 77.53 | 79.69 | 93.55 |
| VGG19 | Gaussian | 60.94 | 49.96 | 60.94 | 52.66 | 52.66 | 60.94 | 92.87 |
| VGG19 | Median | 64.06 | 62.39 | 64.06 | 57.97 | 57.97 | 64.06 | 88.87 |
| VGG19 | Sharpen | 65.62 | 55.07 | 65.62 | 57.88 | 57.88 | 65.62 | 91.08 |
| VGG19 | Sobel | 59.38 | 74.33 | 59.38 | 51.30 | 51.30 | 59.38 | 83.79 |

![VGG19 performance across spatial filters](task02_images/vgg19_overall.png)

## EfficientNet-B0 — All Filters

| Model | Filter | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Macro-F1 (%) | Balanced Accuracy (%) | AUC (%) |
|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | Original | 68.75 | 78.74 | 68.75 | 63.70 | 63.70 | 68.75 | 94.14 |
| EfficientNet-B0 | Mean | 71.88 | 80.14 | 71.88 | 67.54 | 67.54 | 71.88 | 96.42 |
| EfficientNet-B0 | Gaussian | 76.56 | 83.16 | 76.56 | 74.12 | 74.12 | 76.56 | 96.39 |
| EfficientNet-B0 | Median | 78.12 | 83.75 | 78.12 | 75.32 | 75.32 | 78.12 | 94.63 |
| EfficientNet-B0 | Sharpen | 78.12 | 86.34 | 78.12 | 76.96 | 76.96 | 78.12 | 97.92 |
| EfficientNet-B0 | Sobel | 51.56 | 54.70 | 51.56 | 48.12 | 48.12 | 51.56 | 79.26 |

![EfficientNet-B0 performance across spatial filters](task02_images/efficientnet_overall.png)

## AlexNet — All Filters

| Model | Filter | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Macro-F1 (%) | Balanced Accuracy (%) | AUC (%) |
|---|---|---|---|---|---|---|---|---|
| AlexNet | Original | 67.19 | 66.67 | 67.19 | 66.81 | 66.81 | 67.19 | 90.04 |
| AlexNet | Mean | 70.31 | 78.32 | 70.31 | 67.45 | 67.45 | 70.31 | 89.26 |
| AlexNet | Gaussian | 62.50 | 49.13 | 62.50 | 54.08 | 54.08 | 62.50 | 90.23 |
| AlexNet | Median | 65.62 | 62.40 | 65.62 | 60.69 | 60.69 | 65.62 | 87.60 |
| AlexNet | Sharpen | 71.88 | 77.08 | 71.88 | 68.44 | 68.44 | 71.88 | 87.21 |
| AlexNet | Sobel | 53.12 | 43.13 | 53.12 | 45.74 | 45.74 | 53.12 | 73.73 |

![AlexNet performance across spatial filters](task02_images/alexnet_overall.png)

## Final Combined Comparison — All Models × All Filters

| Model | Filter | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Macro-F1 (%) | Balanced Accuracy (%) | AUC (%) |
|---|---|---|---|---|---|---|---|---|
| VGG19 | Original | 70.31 | 72.26 | 70.31 | 67.99 | 67.99 | 70.31 | 94.30 |
| VGG19 | Mean | 79.69 | 84.88 | 79.69 | 77.53 | 77.53 | 79.69 | 93.55 |
| VGG19 | Gaussian | 60.94 | 49.96 | 60.94 | 52.66 | 52.66 | 60.94 | 92.87 |
| VGG19 | Median | 64.06 | 62.39 | 64.06 | 57.97 | 57.97 | 64.06 | 88.87 |
| VGG19 | Sharpen | 65.62 | 55.07 | 65.62 | 57.88 | 57.88 | 65.62 | 91.08 |
| VGG19 | Sobel | 59.38 | 74.33 | 59.38 | 51.30 | 51.30 | 59.38 | 83.79 |
| EfficientNet-B0 | Original | 68.75 | 78.74 | 68.75 | 63.70 | 63.70 | 68.75 | 94.14 |
| EfficientNet-B0 | Mean | 71.88 | 80.14 | 71.88 | 67.54 | 67.54 | 71.88 | 96.42 |
| EfficientNet-B0 | Gaussian | 76.56 | 83.16 | 76.56 | 74.12 | 74.12 | 76.56 | 96.39 |
| EfficientNet-B0 | Median | 78.12 | 83.75 | 78.12 | 75.32 | 75.32 | 78.12 | 94.63 |
| EfficientNet-B0 | Sharpen | 78.12 | 86.34 | 78.12 | 76.96 | 76.96 | 78.12 | 97.92 |
| EfficientNet-B0 | Sobel | 51.56 | 54.70 | 51.56 | 48.12 | 48.12 | 51.56 | 79.26 |
| AlexNet | Original | 67.19 | 66.67 | 67.19 | 66.81 | 66.81 | 67.19 | 90.04 |
| AlexNet | Mean | 70.31 | 78.32 | 70.31 | 67.45 | 67.45 | 70.31 | 89.26 |
| AlexNet | Gaussian | 62.50 | 49.13 | 62.50 | 54.08 | 54.08 | 62.50 | 90.23 |
| AlexNet | Median | 65.62 | 62.40 | 65.62 | 60.69 | 60.69 | 65.62 | 87.60 |
| AlexNet | Sharpen | 71.88 | 77.08 | 71.88 | 68.44 | 68.44 | 71.88 | 87.21 |
| AlexNet | Sobel | 53.12 | 43.13 | 53.12 | 45.74 | 45.74 | 53.12 | 73.73 |

![Final accuracy comparison across all models and filters](task02_images/final_comparison.png)
