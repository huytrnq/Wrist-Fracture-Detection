# Wrist-Fracture-Detection
This repository contains the code for the project "Wrist Fracture Detection" which is a part of the course "Medical Image Analysis" at the University of Cassion.

## Results
### 1. Lightbgm

#### Hog - Results with Hog Feature Extraction
|           | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Class 0   | 0.93      | 0.72   | 0.81     | 1961    |
| Class 1   | 0.39      | 0.77   | 0.52     | 453     |
|           |           |        |          |         |
| Accuracy  |           |        | 0.73     | 2414    |
| Macro avg | 0.66      | 0.74   | 0.66     | 2414    |
| Weighted avg | 0.83  | 0.73   | 0.76     | 2414    |

Confusion Matrix:
|      | Predicted 0 | Predicted 1 |
|------|-------------|-------------|
| True 0 | 1412        | 549         |
| True 1 | 105         | 348         |

F1-score: 0.7563388507935389

#### Hog + Alexnet - Results with Hog and Alexnet Feature Extraction
|           | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Class 0   | 0.93      | 0.73   | 0.82     | 1961    |
| Class 1   | 0.40      | 0.77   | 0.52     | 453     |
|           |           |        |          |         |
| Accuracy  |           |        | 0.74     | 2414    |
| Macro avg | 0.66      | 0.75   | 0.67     | 2414    |
| Weighted avg | 0.83  | 0.74   | 0.76     | 2414    |

Confusion Matrix:
|      | Predicted 0 | Predicted 1 |
|------|-------------|-------------|
| True 0 | 1433        | 528         |
| True 1 | 105         | 348         |

F1-score: 0.7636598187819055

#### Hog + Alexnet + Hog-Canny - Results with Hog, Alexnet and Canny Feature Extraction
|           | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Class 0   | 0.94      | 0.73   | 0.82     | 1961    |
| Class 1   | 0.40      | 0.78   | 0.53     | 453     |
|           |           |        |          |         |
| Accuracy  |           |        | 0.74     | 2414    |
| Macro avg | 0.67      | 0.76   | 0.68     | 2414    |
| Weighted avg | 0.84  | 0.74   | 0.77     | 2414    |

Confusion Matrix:
|      | Predicted 0 | Predicted 1 |
|------|-------------|-------------|
| True 0 | 1435        | 526         |
| True 1 | 98          | 355         |

F1-score: 0.7671430358179476

#### Hog + Alexnet + Hog-Canny - Results with Hog, Alexnet and Canny Feature Extraction + HardMining
|           | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Class 0   | 0.94      | 0.75   | 0.84     | 1961    |
| Class 1   | 0.42      | 0.79   | 0.55     | 453     |
|           |           |        |          |         |
| Accuracy  |           |        | 0.76     | 2414    |
| Macro avg | 0.68      | 0.77   | 0.69     | 2414    |
| Weighted avg | 0.84  | 0.76   | 0.78     | 2414    |

Confusion Matrix:
|      | Predicted 0 | Predicted 1 |
|------|-------------|-------------|
| True 0 | 1473        | 488         |
| True 1 | 94          | 359         |

F1-score: 0.7819789085731801
