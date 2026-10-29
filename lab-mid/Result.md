# Complete Results: Blood Cell Classification (HOG vs SIFT + Linear SVM)

*All important details and results from `Untitled12.ipynb`, with the plots. Keep this file and the `images/` folder together.*

---

## 1. Experiment Setup

| Item | Value |
|---|---|
| Roll number | 034 |
| **Random seed** | **34** (last two digits of roll number) |
| Dataset | BloodMNIST (MedMNIST) |
| Original image size | 64 x 64, RGB (3 channels) |
| Classes used (label IDs) | 1, 5, 6, 7 |
| Class names | Eosinophil, Monocyte, Neutrophil, Platelet |
| Classifier | LinearSVC (one-vs-rest), max_iter = 20000, random_state = 34 |
| Scaling | StandardScaler (fit on train only) |
| C values tried | 0.1, 1, 10 |
| Random-guess baseline | 25% (4 balanced classes) |

## 2. Dataset Sizes

| Split | Full BloodMNIST shape | Images used | Per class | After preprocessing |
|---|---|---|---|---|
| Train | (11959, 64, 64, 3) | 400 | 100 | (400, 128, 128) |
| Validation | (1712, 64, 64, 3) | 120 | 30 | (120, 128, 128) |
| Test | (3421, 64, 64, 3) | 120 | 30 | (120, 128, 128) |

Every class has exactly the same number of images in every split (Eosinophil, Monocyte, Neutrophil, Platelet).

**Preprocessing:** RGB to grayscale, then resize 64x64 to 128x128.

![Sample images](images/01_sample_images.png)

## 3. Feature Extraction Details

### HOG
| Parameter | Value |
|---|---|
| Orientations | 9 |
| Pixels per cell | 8 x 8 (16 x 16 = 256 cells per image) |
| Cells per block | 2 x 2 (15 x 15 = 225 blocks) |
| Values per block | 2 x 2 x 9 = 36 |
| **HOG feature shape (train)** | **(400, 8100)** |
| **Feature length** | **8100** (225 x 36) |

![HOG visualisation](images/02_hog_visualisation.png)

### SIFT
**Normal keypoint detection (exploration, training images):**

| Statistic | Value |
|---|---|
| Average keypoints per image | 72.2975 |
| Minimum | 9 |
| Maximum | 178 |
| Images with zero keypoints | 0 |

![SIFT keypoints](images/04_sift_keypoints.png)

**Dense SIFT used for training:** 4 x 4 grid at x, y = 16, 48, 80, 112; keypoint size 32; 16 descriptors x 128 values.

| Parameter | Value |
|---|---|
| **SIFT feature shape (train)** | **(400, 2048)** |
| **Feature length** | **2048** (16 x 128) |

![SIFT grid](images/05_sift_grid.png)

### Comparison
| | HOG | SIFT |
|---|---|---|
| Feature length | 8100 | 2048 |
| Train feature matrix | 400 x 8100 | 400 x 2048 |
| Best C | 0.1 | 1 |

## 4. C Tuning (Validation Accuracy)

| Features | C = 0.1 | C = 1 | C = 10 | Best C |
|---|---|---|---|---|
| HOG | 0.7000 | 0.7000 | 0.7000 | 0.1 (tie, first kept) |
| SIFT | 0.7333 | 0.7500 | 0.7417 | 1 |

## 5. Main Results (Validation Set, 120 images)

| Metric | HOG | SIFT |
|---|---|---|
| **Accuracy** | **0.7000 (70%)** | **0.7500 (75%)** |
| Correct / total | 84 / 120 | 90 / 120 |
| Wrong predictions | 36 | 30 |
| **Macro F1** | **0.6971** | **0.7493** |

### HOG confusion matrix
![HOG confusion matrix](images/03_hog_confusion_matrix.png)

| Actual \ Predicted | Eosinophil | Monocyte | Neutrophil | Platelet | Row total |
|---|---|---|---|---|---|
| **Eosinophil** | **17** | 7 | 6 | 0 | 30 |
| **Monocyte** | 5 | **22** | 3 | 0 | 30 |
| **Neutrophil** | 7 | 8 | **15** | 0 | 30 |
| **Platelet** | 0 | 0 | 0 | **30** | 30 |

**HOG per-class results (one-vs-rest positives and negatives):**

| Class | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Eosinophil | 17 | 12 | 13 | 78 | 0.5862 | 0.5667 | 0.5763 |
| Monocyte | 22 | 15 | 8 | 75 | 0.5946 | 0.7333 | 0.6567 |
| Neutrophil | 15 | 9 | 15 | 81 | 0.6250 | 0.5000 | 0.5556 |
| Platelet | 30 | 0 | 0 | 90 | 1.0000 | 1.0000 | 1.0000 |
| **Macro average** | | | | | | | **0.6971** |

### SIFT confusion matrix
![SIFT confusion matrix](images/06_sift_confusion_matrix.png)

| Actual \ Predicted | Eosinophil | Monocyte | Neutrophil | Platelet | Row total |
|---|---|---|---|---|---|
| **Eosinophil** | **18** | 10 | 1 | 1 | 30 |
| **Monocyte** | 7 | **23** | 0 | 0 | 30 |
| **Neutrophil** | 4 | 3 | **20** | 3 | 30 |
| **Platelet** | 0 | 0 | 1 | **29** | 30 |

**SIFT per-class results (one-vs-rest positives and negatives):**

| Class | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Eosinophil | 18 | 11 | 12 | 79 | 0.6207 | 0.6000 | 0.6102 |
| Monocyte | 23 | 13 | 7 | 77 | 0.6389 | 0.7667 | 0.6970 |
| Neutrophil | 20 | 2 | 10 | 88 | 0.9091 | 0.6667 | 0.7692 |
| Platelet | 29 | 4 | 1 | 86 | 0.8788 | 0.9667 | 0.9206 |
| **Macro average** | | | | | | | **0.7493** |

**Positive / negative meaning:** for each class, that class is "positive" and all others "negative". TP = correctly found, FP = false alarm (other cell called this class), FN = missed, TN = correctly rejected. Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = harmonic mean of both.

## 6. Most Confused Classes

Combined HOG + SIFT errors per pair (both directions):

| Pair | HOG | SIFT | Combined |
|---|---|---|---|
| Eosinophil vs Monocyte | 12 | 17 | **29** |
| Eosinophil vs Neutrophil | 13 | 5 | **18** |
| Monocyte vs Neutrophil | 11 | 3 | **14** |
| Neutrophil vs Platelet | 0 | 4 | **4** |
| Eosinophil vs Platelet | 0 | 1 | **1** |
| Monocyte vs Platelet | 0 | 0 | **0** |

**Output from notebook:** `Most confused classes: ('Eosinophil', 'Monocyte', 29)`

## 7. Robustness Test (Test Set, 120 images, no retraining)

| Test condition | Setting |
|---|---|
| Rotated | 20 degrees about centre (64, 64), scale 1 |
| Noisy | Gaussian noise, mean 0, std 0.05 (on 0-1 scale), seed 34 |

![Clean, rotated, noisy](images/07_clean_rotated_noisy.png)

| Model | Clean | Rotated | Noisy |
|---|---|---|---|
| HOG | 0.7750 | 0.7000 | 0.2500 |
| SIFT | 0.7417 | 0.3333 | 0.6917 |

| Model | Drop from clean to rotated | Drop from clean to noisy |
|---|---|---|
| HOG | -0.0750 | **-0.5250** |
| SIFT | **-0.4083** | -0.0500 |

![Robustness bar chart](images/08_robustness_bar_chart.png)

## 8. Explanation of Results

- **Accuracy:** HOG reached 70% and SIFT 75% on validation, both far above the 25% random baseline. On clean test images HOG scored 77.5% and SIFT 74.2%, so the two are close; differences of a few percent are only a few images out of 120.
- **Platelets** are classified almost perfectly (30/30 HOG, 29/30 SIFT) because they are small and have no nucleus.
- **Eosinophil, Monocyte and Neutrophil** are confused with each other; Eosinophil vs Monocyte is the worst pair (29 combined errors).
- **HOG** is weak on noise (25% = chance) because noise creates false edge gradients, but handles rotation fairly well.
- **SIFT** is robust to noise but fails on rotation (33%) because descriptors were taken on a fixed grid without orientation.

## 9. Final Summary

| Question | Answer |
|---|---|
| Better on validation? | SIFT (75% vs 70%) |
| Better on clean test? | HOG (77.5% vs 74.2%) |
| Better under rotation? | HOG (70% vs 33%) |
| Better under noise? | SIFT (69% vs 25%) |
| Easiest class | Platelet |
| Hardest pair | Eosinophil vs Monocyte |
