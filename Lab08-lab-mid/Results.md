# Lab Mid Exam: HOG vs SIFT on Blood-Cell Images — Results

**Name:** Pakiza Hassan  
**Reg Number:** FA23-BAI-034

## Personal Settings

| Setting | Rule | My value |
|---|---|---|
| Random seed | Last two digits of registration number `034` | **34** |
| HOG cell size | Last digit `4` is even | **8 × 8** |
| SIFT grid size | Last digit `4` is between 0 and 4 | **4 × 4 grid (16 keypoints)** |

---

## Q1. Data Setup (3 marks)

**Dataset:** BloodMNIST (64 × 64 RGB images)  
**Selected classes:** Eosinophil (1), Monocyte (5), Neutrophil (6), Platelet (7)

**Full dataset size**

| Split | Images | Labels |
|---|---|---|
| Train | (11959, 64, 64, 3) | (11959,) |
| Validation | (1712, 64, 64, 3) | (1712,) |
| Test | (3421, 64, 64, 3) | (3421,) |

**Selected samples: 100 training, 30 validation and 30 test images per class**

| Class | Class ID | Train | Validation | Test |
|---|---|---|---|---|
| Eosinophil | 1 | 100 | 30 | 30 |
| Monocyte | 5 | 100 | 30 | 30 |
| Neutrophil | 6 | 100 | 30 | 30 |
| Platelet | 7 | 100 | 30 | 30 |
| **Total** | | **400** | **120** | **120** |

**Grayscale conversion and resize to 128 × 128**

| Split | Shape |
|---|---|
| Train grayscale | (400, 128, 128) |
| Validation grayscale | (120, 128, 128) |
| Test grayscale | (120, 128, 128) |

![Sample images from each class](images/01_sample_cell_images.png)

---

## Q2. HOG + SVM (6 marks)

HOG training feature matrix: (400, 8100)  
HOG validation feature matrix: (120, 8100)  
HOG feature length: **8100**

### Q2.1 — HOG feature-length calculation

- Cells per side: 128 / 8 = 16
- Block positions per side (2 × 2 cells per block): 16 − 2 + 1 = 15
- Values per block: 2 × 2 × 9 = 36
- Feature length: 15 × 15 × 36 = **8100 features**

**HOG visualization for one image per class**

![HOG visualization](images/02_hog_visualization.png)

**Train HOG + SVM and choose C using validation data**

| C | Validation accuracy | Validation macro-F1 |
|---|---|---|
| 0.1 | 0.6917 | 0.6882 |
| 1 | 0.6917 | 0.6882 |
| 10 | 0.6917 | 0.6882 |

Selected **C = 0.1**.

**HOG + SVM validation results**

Validation accuracy: **69.17%** | Validation macro-F1: **68.82%**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Eosinophil | 0.58 | 0.47 | 0.52 | 30 |
| Monocyte | 0.56 | 0.73 | 0.64 | 30 |
| Neutrophil | 0.63 | 0.57 | 0.60 | 30 |
| Platelet | 1.00 | 1.00 | 1.00 | 30 |
| Accuracy | | | 0.69 | 120 |
| Macro avg | 0.69 | 0.69 | 0.69 | 120 |
| Weighted avg | 0.69 | 0.69 | 0.69 | 120 |

![HOG + SVM validation confusion matrix](images/03_hog_svm_confusion_matrix.png)

---

## Q3. SIFT + SVM (7 marks)

**Natural SIFT keypoints (training images)**

| Statistic | Value |
|---|---|
| Average keypoints per image | 72.3 |
| Minimum keypoints | 9 |
| Maximum keypoints | 178 |
| Images with zero keypoints | 0 |

![Natural SIFT keypoints on one image](images/04_natural_sift_keypoints.png)

**Q3 explanation:** Natural SIFT produces a different number of keypoints for different images, but a standard SVM expects every input feature vector to have the same length.

**Fixed grid of SIFT keypoints**

The grid has 4 × 4 = 16 keypoints with keypoint size 32. SIFT training feature matrix: (400, 2048). SIFT validation feature matrix: (120, 2048).

![Fixed 4 × 4 SIFT grid](images/05_sift_4x4_grid.png)

### Q3.2 — SIFT feature-length calculation

The SIFT grid is 4 × 4, so it has 16 keypoints. Each descriptor has 128 values. Feature length: 16 × 128 = **2048 features**.

**Train SIFT + SVM and choose C using validation data**

| C | Validation accuracy | Validation macro-F1 |
|---|---|---|
| 0.1 | 0.7250 | 0.7234 |
| 1 | 0.7333 | 0.7345 |
| 10 | 0.7333 | 0.7345 |

Selected **C = 1**.

**SIFT + SVM validation results**

Validation accuracy: **73.33%** | Validation macro-F1: **73.45%**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Eosinophil | 0.64 | 0.53 | 0.58 | 30 |
| Monocyte | 0.59 | 0.77 | 0.67 | 30 |
| Neutrophil | 0.75 | 0.70 | 0.72 | 30 |
| Platelet | 1.00 | 0.93 | 0.97 | 30 |
| Accuracy | | | 0.73 | 120 |
| Macro avg | 0.74 | 0.73 | 0.73 | 120 |
| Weighted avg | 0.74 | 0.73 | 0.73 | 120 |

![SIFT + SVM validation confusion matrix](images/06_sift_svm_confusion_matrix.png)

**Validation comparison**

| Model | Chosen C | Validation accuracy | Validation macro-F1 | Feature length |
|---|---|---|---|---|
| HOG + SVM | 0.1 | 0.6917 | 0.6882 | 8100 |
| SIFT + SVM | 1 | 0.7333 | 0.7345 | 2048 |

---

## Q4. Robustness (6 marks)

### Q4.1 Prediction

- **Rotation prediction:** HOG may lose more accuracy under rotation because rotating an image changes the gradient directions summarized by HOG.
- **Noise prediction:** HOG may lose more accuracy under noise because noise can add unwanted gradients to the image.

**Rotated and noisy versions of the test images**

Rotation is 20 degrees. Gaussian noise uses sigma = 0.05 on pixels scaled to 0–1. The images are transformed before feature extraction. Clean, rotated and noisy test images: (120, 128, 128) each.

![Clean, rotated and noisy test images](images/07_clean_rotated_noisy_examples.png)

### Q4.2 — Results table

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
|---|---:|---:|---:|
| HOG + SVM | 71.67% | 63.33% | 25.00% |
| SIFT + SVM | 75.83% | 31.67% | 70.83% |

| Model | Rotation accuracy drop | Noise accuracy drop |
|---|---:|---:|
| HOG + SVM | 8.33% | 46.67% |
| SIFT + SVM | 44.17% | 5.00% |

![HOG vs SIFT test-set robustness](images/08_robustness_bar_chart.png)

### Q4.3 — Was your prediction correct?

1. Under rotation, **SIFT + SVM** lost more accuracy: 75.83% to 31.67%, a decrease of **44.17 percentage points**.
2. Under noise, **HOG + SVM** lost more accuracy: 71.67% to 25.00%, a decrease of **46.67 percentage points**.
3. Therefore, my prediction was **partly correct** based on the measured results.

---

## Q5. Short Answers and AI Log (3 marks)

**Most-confused pair (combined validation confusion matrices)**

| Class 1 | Class 2 | Combined cross-confusion |
|---|---|---|
| Eosinophil | Monocyte | 30 |
| Eosinophil | Neutrophil | 19 |
| Monocyte | Neutrophil | 18 |
| Neutrophil | Platelet | 2 |
| Eosinophil | Platelet | 0 |
| Monocyte | Platelet | 0 |

### Q5.1 (a) Why does HOG use block normalisation?

HOG uses block normalisation to reduce the effects of lighting and contrast changes. It makes gradient features more consistent across image regions and can improve classification accuracy.

### Q5.1 (b) Why is it wrong to choose C using the test set?

The test set should be reserved for the final evaluation of the model on unseen data. Choosing C using test-set performance leaks information from the test set and may lead to an overly optimistic evaluation. Therefore, C should be selected using the validation set.

### Q5.2 Which two classes did your models confuse most, and why might grayscale make this harder?

The two classes confused most were **Eosinophil and Monocyte**, with a combined cross-confusion count of **30** according to the validation confusion matrices. These classes may have similar visual features, making them difficult for the models to distinguish. Grayscale removes colour information, including staining differences, so the models must rely mainly on shape, texture, and gradient features to classify the cells.

### AI usage log

| Prompt (one line) | What I changed | How I checked |
|---|---|---|
| Help me organize a HOG and SIFT BloodMNIST classification notebook. | Set registration-specific seed, HOG cell size, and SIFT grid size. | Printed the settings, data shapes, and class counts. |
| Explain how to extract fixed-length SIFT features and train an SVM. | Used a fixed 4 x 4 keypoint grid and validation-based C selection. | Printed feature matrix shapes and feature length; checked validation metrics. |
| Help test HOG and SIFT under image rotation and Gaussian noise. | Applied 20-degree rotation and Gaussian noise before feature extraction. | Viewed transformed images and compared test accuracies in the output table. |
