# SIFT Landmark Recognition — Results

**Method:** SIFT keypoints (up to 2,500 per image, 128-D descriptors) → brute-force L2 matching with Lowe's ratio test → RANSAC homography verification → landmark ranking.

## Dataset Overview

| Landmark | Image Files | Readable Images |
|---|---|---|
| christ_the_reedemer | 516 | 516 |
| eiffel_tower | 624 | 624 |
| great_wall_of_china | 626 | 626 |
| roman_colosseum | 630 | 630 |
| taj_mahal | 252 | 252 |
| **Total** | **2,648** | **2,648** |

![Images per category](lab06_images/images_per_category.png)

## SIFT Feature Extraction

Usable reference images: **678**, grouped into 6 landmark classes with descriptors (the five above plus an `unknown` class).

| Landmark | Mean Keypoints | Median | Min | Max | Reference Images |
|---|---|---|---|---|---|
| christ_the_reedemer | 979.9 | 547.0 | 58 | 2501 | 130 |
| eiffel_tower | 1207.6 | 670.0 | 60 | 2501 | 158 |
| great_wall_of_china | 1336.9 | 740.0 | 32 | 2502 | 158 |
| roman_colosseum | 1219.0 | 763.0 | 255 | 2501 | 158 |
| taj_mahal | 1128.1 | 483.5 | 246 | 2500 | 64 |
| unknown | 1251.8 | 1136.0 | 233 | 2500 | 10 |

![Keypoints per landmark](lab06_images/keypoints_per_landmark.png)

### Detected Keypoints (Examples)

![Keypoints example 1](lab06_images/keypoints_example_1.png)

![Keypoints example 2](lab06_images/keypoints_example_2.png)

![Keypoints example 3](lab06_images/keypoints_example_3.png)

![Keypoints example 4](lab06_images/keypoints_example_4.png)

![Keypoints example 5](lab06_images/keypoints_example_5.png)

## Query Image

Query image `0016510b5c.jpg` — 828 × 466 px, **1,136** keypoints.

![Query image](lab06_images/query_image.png)

## Feature Matching and Landmark Ranking

| Rank | Landmark | Score | Best Reliable Matches | Reference Images |
|---|---|---|---|---|
| 1 | unknown | 1372.6 | 1136 | 10 |
| 2 | eiffel_tower | 110.4 | 83 | 158 |
| 3 | roman_colosseum | 94.4 | 69 | 158 |
| 4 | great_wall_of_china | 92.2 | 69 | 158 |
| 5 | christ_the_reedemer | 83.2 | 60 | 130 |
| 6 | taj_mahal | 76.2 | 57 | 64 |

![Candidate ranking](lab06_images/candidate_ranking.png)

## RANSAC Homography and Localization

| Metric | Value |
|---|---|
| Predicted landmark | unknown |
| Reliable (Lowe-ratio) matches | 1136 |
| RANSAC inliers | 1136 |
| Inlier ratio | 100.0% |
| Geometry verified | True |
| Decision | RECOGNIZED |

![Matched features](lab06_images/matched_features.png)

![Landmark localization](lab06_images/localization.png)

## Experimental Evaluation

Four test conditions were applied to the query image.

![Evaluation images](lab06_images/evaluation_images.png)

| Test Condition | Predicted Landmark | Keypoints | Reliable Matches | RANSAC Inliers | Inlier Ratio | Geometry Verified | Decision | Processing Time (s) |
|---|---|---|---|---|---|---|---|---|
| Original | unknown | 1136 | 1136 | 1136 | 1.000 | True | Recognized | 34.67 |
| Rotated (20°) | unknown | 1264 | 654 | 613 | 0.937 | True | Recognized | 44.54 |
| Resized (65%) | unknown | 547 | 368 | 358 | 0.973 | True | Recognized | 15.39 |
| Reduced brightness (50%) | unknown | 617 | 582 | 579 | 0.995 | True | Recognized | 17.45 |

![Keypoints per test](lab06_images/keypoints_per_test.png)

![Reliable matches per test](lab06_images/reliable_matches_per_test.png)

![RANSAC inliers per test](lab06_images/ransac_inliers_per_test.png)

![Processing time per test](lab06_images/processing_time_per_test.png)

### Correctness

| Test Condition | Expected Label | Predicted Landmark | Correct |
|---|---|---|---|
| Original | Badshahi Mosque | unknown | False |
| Rotated (20°) | Badshahi Mosque | unknown | False |
| Resized (65%) | Badshahi Mosque | unknown | False |
| Reduced brightness (50%) | Badshahi Mosque | unknown | False |

**Evaluated images:** 4 | **Correct predictions:** 0 | **Labeled accuracy:** 0.00%

![Prediction correctness](lab06_images/prediction_correctness.png)

## Notes on These Results

- The query image matched the `unknown` reference class with all 1,136 keypoints (100% inliers), which suggests the query image itself is in that reference set rather than being a genuine independent match.
- The expected label, "Badshahi Mosque", is not one of the five landmark classes in the reference database, so the 0% accuracy reflects a label/database mismatch rather than a measured SIFT failure.
- The recognition scores were consistent across rotation, resizing, and brightness changes (inlier ratios 0.94–1.00), which does show SIFT's robustness to those transformations.
