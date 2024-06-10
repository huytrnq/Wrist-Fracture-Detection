# Wrist-Fracture-Detection
This repository contains the code for the project "Wrist Fracture Detection" which is a part of the course "Medical Image Analysis" at the University of Cassion.

## Image Preprocessing Pipeline
![image](./images/Image%20Processing%20Pipeline.png)

## Feature Extraction 

## Model Training
### Moddel Selection
- LightGBM
### Training Pipeline
- HardMining

## Results
### 1. Lightbgm

#### Hog - Results with Hog Feature Extraction

    |           | Precision | Recall | F1-score | Support |
    |-----------|-----------|--------|----------|---------|
    | Class 0   | 0.94      | 0.73   | 0.82     | 1996    |
    | Class 1   | 0.40      | 0.78   | 0.53     | 453     |
    |           |           |        |          |         |
    | Accuracy  |           |        | 0.74     | 2449    |
    | Macro avg | 0.67      | 0.75   | 0.67     | 2449    |
    | Weighted avg | 0.84  | 0.74   | 0.77     | 2449    |

    Confusion Matrix:
    |      | Predicted 0 | Predicted 1 |
    |------|-------------|-------------|
    | True 0 | 1461        | 535         |
    | True 1 | 101         | 352         |

    F1-score: 0.7665189148349725

#### Hog + Alexnet - Results with Hog and Alexnet Feature Extraction

    |           | Precision | Recall | F1-score | Support |
    |-----------|-----------|--------|----------|---------|
    | Class 0   | 0.93      | 0.75   | 0.83     | 1996    |
    | Class 1   | 0.40      | 0.75   | 0.52     | 453     |
    |           |           |        |          |         |
    | Accuracy  |           |        | 0.75     | 2449    |
    | Macro avg | 0.67      | 0.75   | 0.68     | 2449    |
    | Weighted avg | 0.83  | 0.75   | 0.77     | 2449    |

    Confusion Matrix:
    |      | Predicted 0 | Predicted 1 |
    |------|-------------|-------------|
    | True 0 | 1488        | 508         |
    | True 1 | 111         | 342         |

    F1-score: 0.7717927657310995

#### Hog + Alexnet + Hog-Canny - Results with Hog, Alexnet and Canny Feature Extraction

    |           | Precision | Recall | F1-score | Support |
    |-----------|-----------|--------|----------|---------|
    | Class 0   | 0.94      | 0.75   | 0.83     | 1996    |
    | Class 1   | 0.42      | 0.78   | 0.54     | 453     |
    |           |           |        |          |         |
    | Accuracy  |           |        | 0.76     | 2449    |
    | Macro avg | 0.68      | 0.76   | 0.69     | 2449    |
    | Weighted avg | 0.84  | 0.76   | 0.78     | 2449    |

    Confusion Matrix:
    |      | Predicted 0 | Predicted 1 |
    |------|-------------|-------------|
    | True 0 | 1500        | 496         |
    | True 1 | 101         | 352         |

    F1-score: 0.7798484946093364

#### Hog + Alexnet + Hog-Canny - Results with Hog, Alexnet and Canny Feature Extraction + HardMining
|           | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Class 0   | 0.94      | 0.75   | 0.84     | 1996    |
| Class 1   | 0.42      | 0.79   | 0.55     | 453     |
|           |           |        |          |         |
| Accuracy  |           |        | 0.76     | 2449    |
| Macro avg | 0.68      | 0.77   | 0.69     | 2449    |
| Weighted avg | 0.84  | 0.76   | 0.78     | 2449    |

Confusion Matrix:
|      | Predicted 0 | Predicted 1 |
|------|-------------|-------------|
| True 0 | 1499        | 497         |
| True 1 | 94          | 359         |

F1-score: 0.7822761136208711
