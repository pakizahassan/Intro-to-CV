# Key Results: Blood Cell Classification (HOG vs SIFT + Linear SVM)

## 1. Important Settings

| Item | Value |
|---|---|
| Seed | 34 (from roll number 034) |
| HOG cell | (8, 8) pixels |
| SIFT grid | 4 x 4 (16 keypoints per image) |
| Classes (label IDs) | [1, 5, 6, 7] |
| Class names | Eosinophil, Monocyte, Neutrophil, Platelet |
| Image size after preprocessing | 128 x 128 grayscale |
| Classifier | LinearSVC with StandardScaler |

```python
HOG_CELL = (8, 8)
SIFT_GRID = 4
CLASSES = [1, 5, 6, 7]
NAMES = ["Eosinophil", "Monocyte", "Neutrophil", "Platelet"]
```

## 2. Data Used

| Split | Images | Per class |
|---|---|---|
| Train | 400 | 100 |
| Validation | 120 | 30 |
| Test | 120 | 30 |

## 3. Feature Length

| Feature | Shape (train) | Feature length | How it is calculated |
|---|---|---|---|
| HOG | (400, 8100) | **8100** | 15 x 15 blocks x (2 x 2 x 9) = 225 x 36 |
| SIFT | (400, 2048) | **2048** | 16 grid points x 128 values |

## 4. Accuracy and F1

| Model | Best C | Validation accuracy | Macro F1 | Clean test accuracy |
|---|---|---|---|---|
| HOG | 0.1 | **70% (84/120)** | 0.6971 | 77.5% |
| SIFT | 1 | **75% (90/120)** | 0.7493 | 74.2% |

Random guessing would give 25%, so both models learn well.

## 5. Confusion Matrices (Validation)

**HOG**

![HOG confusion matrix](images/03_hog_confusion_matrix.png)

**SIFT**

![SIFT confusion matrix](images/06_sift_confusion_matrix.png)

- Platelet is almost always correct (30/30 HOG, 29/30 SIFT).
- Most confused pair: **Eosinophil vs Monocyte** (29 combined errors).

## 6. Robustness (Test Accuracy)

| Model | Clean | Rotated (20°) | Noisy |
|---|---|---|---|
| HOG | 77.5% | 70.0% | 25.0% |
| SIFT | 74.2% | 33.3% | 69.2% |

![Robustness](images/08_robustness_bar_chart.png)

## 7. Explanation of the Results

**Why accuracy is 70% and 75%.** The model must choose between 4 cell types, so random guessing would be right only 25% of the time. HOG (70%) and SIFT (75%) are about three times better than guessing. SIFT got 90 of 120 validation images right and HOG got 84. With only 120 images, the 5-point gap is just 6 images, so the two methods are close.

**Why HOG and SIFT give different lengths.** HOG describes edge directions across the whole image (225 blocks x 36 values = 8100). SIFT describes only 16 fixed patches (16 x 128 = 2048). HOG is more detailed but larger; SIFT is smaller and more compact.

**Why Platelets are easy.** Platelets are small and have no nucleus, so they look very different from the three white blood cells. Both models get them almost 100% right.

**Why Eosinophil, Monocyte and Neutrophil are confused.** These three are white blood cells of similar size with a nucleus. Grayscale HOG and SIFT features capture shape and texture, but not the colour of the granules, which is a main cue doctors use. Eosinophil vs Monocyte is the hardest pair (29 combined errors).

**Positive and negative.** For each class, that class counts as "positive" and all others as "negative". A *true positive* is a correctly found cell, a *false negative* is a missed cell, a *false positive* is another cell wrongly called this class, and a *true negative* is a correctly rejected cell. Example (Eosinophil, HOG): TP = 17, FN = 13, FP = 12, TN = 78.

**Why HOG fails with noise (25%).** HOG is built from edge gradients. Random noise creates many false edges, so the features no longer describe the cell. 25% is the same as random guessing. HOG handles rotation better (70%) because the cells are roughly round and the model still sees similar edge patterns.

**Why SIFT fails with rotation (33%).** Normal SIFT is rotation invariant, but here descriptors were taken at fixed grid points with fixed orientation. When the image rotates 20 degrees, each grid point sees a different part of the cell, so the features change. SIFT handles noise well (69%) because each descriptor averages gradients over a large 32-pixel patch, which smooths out small noise.

**Why the models were not retrained for the robustness test.** The goal was to see how a model trained on clean images behaves on damaged images. Training on rotated and noisy images (data augmentation) would likely improve these results.

## 8. Conclusion

- SIFT is better on validation (75% vs 70%); HOG is slightly better on clean test (77.5% vs 74.2%).
- HOG fails with noise; SIFT (fixed grid) fails with rotation, so neither is best in every situation.
- Platelets are easy; Eosinophil vs Monocyte is the hardest pair.
- Possible improvements: use colour, data augmentation, denoising before HOG, oriented SIFT keypoints, and more training images.
