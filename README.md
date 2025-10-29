# iris-lassification

## Project Overview
This project explores the **Iris dataset** and the goal is to classify iris flowers into one of three species (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*) based on four measurements:

- Sepal length  
- Sepal width  
- Petal length 
- Petal width

## EDA
- Viewing feature distributions using histograms
- Generating scatterplots to observe separability between classes

## Data Preprocessing
1. **Label Encoding**  
   Species names were converted to numeric labels using `LabelEncoder` (fitted once on all labels to maintain consistent mapping).

2. **Feature Scaling**  
   Applied `StandardScaler` to ensure all features have zero mean and unit variance — important for distance-based algorithms like kNN.

3. **Train-Test Split**  
   - 70% training, 30% testing  
   - Stratified by species to preserve class proportions  


## Model Training
Three classification algorithms were tested:
- **k-Nearest Neighbors (kNN)**
- **Support Vector Machine (SVM)**
- **Random Forest**

## Evaluation and Results
| Model | Accuracy | Notes |
|------:|:-------:|-------|
| kNN   | 0.9778  | One boundary error (Versicolor ↔ Virginica) |
| SVM   | 1.0000  | Perfect on this split with scaling |
| RF    | 1.0000  | Perfect on this split; robust boundaries |

## Analysis and Interpretation

- **SVM** and **Random Forest** achieved **100% accuracy** on the 70/30 stratified split.  
  This is consistent with Iris’ well-separated petal features and proper scaling.
- **kNN** reached **97.78%**, with a single misclassification on the known Versicolor–Virginica overlap region.
- EDA insight holds: **petal length/width** dominate separability; **sepal width** overlaps more and contributes less.

## Learnings

### ML Workflow
- Realized that **stratified sampling** preserves class proportions, ensuring a fairer train-test split.

### Data Preprocessing
- Learnt why **categorical labels** must be converted into numerical form using `LabelEncoder`.
- Understood that fitting `LabelEncoder` on all labels (not just training) is **safe**, since it only learns label mappings, not data patterns.
- Grasped that **feature scaling** (via `StandardScaler`) standardises all features to have zero mean and unit variance.
- Scaling is essential for distance-based algorithms like **kNN** and margin-based models like **SVM**.
    - For **kNN**, the algorithm measures distances (typically Euclidean) between points.  
    If one feature (e.g. *petal length* in cm) has a much larger numerical range than another (e.g. *sepal width* in mm),  
    that feature will **dominate the distance calculation**, even if it’s not more important.  
    Standardizing features to the same scale ensures that all contribute equally to the distance metric.
  - For **SVM**, the algorithm tries to find the best separating hyperplane by maximizing the margin between classes.  
    If features have different magnitudes, the hyperplane’s orientation and margin width become **biased** toward features  
    with larger numerical values, leading to distorted decision boundaries.  
    Scaling normalizes feature influence so that SVM computes margins correctly and converges faster during training.
- Recognized that data preprocessing steps (e.g. scaling) should be fitted on the **training data only**, then applied (transformed) on the test data to prevent data leakage.

### Models

- **k-Nearest Neighbors (kNN)**  
  kNN is a **non-parametric, instance-based** algorithm. It doesn’t learn an explicit model during training — instead, it stores all the training data.  
  When a new sample is introduced, kNN looks for the *k* closest samples (neighbors) in the training set based on a distance metric (usually **Euclidean distance**).  
  The predicted class is then determined by a **majority vote** among these neighbors.  
  > **“Look around me — what do my neighbors say?”**

---

- **Support Vector Machine (SVM)**  
  SVM is a **margin-based classifier** that finds the **optimal hyperplane** separating classes in feature space.  
  It maximizes the **margin** — the distance between the hyperplane and the nearest data points (called **support vectors**).  
  When data isn’t linearly separable, SVM can project it into a higher-dimensional space using **kernels** (e.g. RBF or polynomial) to find a linear separation there.  
  > **“Find the widest possible boundary that separates species.”**

---

- **Random Forest (RF)**  
  Random Forest is an **ensemble learning method** based on **Decision Trees**.  
  It builds many individual trees, each trained on a random subset of the data and features (via **bagging** and **feature randomness**).  
  Each tree makes a prediction, and the final output is decided by **majority vote** across all trees.  
  > **“Let many decision trees vote to reduce individual bias.”**
 

### Model Evaluation
- **Accuracy** — overall correctness.
- **Precision** — proportion of correct positive predictions.
- **Recall** — proportion of actual positives correctly identified.
- **F1-score** — harmonic mean of precision and recall for balanced performance.
- F1-score is more informative than accuracy when balancing false positives and false negatives.

### Concepts
- Differentiated between:
  - `fit()`, `transform()`, and `fit_transform()`  
  - `transform()` and `inverse_transform()`  
- **`fit_transform()`** performs both fitting and transformation in one step, and still allows access to `classes_`.
- Understood why calling `fit()` on the entire label set does not cause data leakage.
- Grasped the difference between **training-time fitting** (for scalers and models) and **encoding-time fitting** (for labels).
- Clarified that **data leakage** only occurs when statistical information from the test set influences the training phase.

