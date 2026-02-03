# Heart Disease ML Prediction 🫀

## Overview

This project implements machine learning algorithms to predict heart disease (cardiovascular disease - CVD) using clinical patient data. Early detection of high-risk individuals plays a vital role in managing and treating cardiovascular diseases, which are the leading cause of death worldwide.

This project was developed as part of an Artificial Intelligence course assignment, focusing on implementing and comparing **Naive Bayes** and **Decision Tree** classification algorithms, along with ensemble methods like **Random Forest** and **XGBoost**.

## 🎯 Objectives

- Develop intelligent models using Naive Bayes and Decision Tree algorithms
- Predict the probability of heart failure in patients
- Perform comprehensive data preprocessing and feature engineering
- Evaluate and compare multiple machine learning models
- Provide detailed technical justification for all decisions made

## 📊 Dataset Description

The dataset contains medical records of patients with the following features:

### Features

| Feature | Type | Description |
|---------|------|-------------|
| **Age** | Numerical | Patient age in years |
| **Sex** | Categorical | Gender (M: Male, F: Female) |
| **ChestPainType** | Categorical | Type of chest pain (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic) |
| **RestingBP** | Numerical | Resting blood pressure (mm Hg) |
| **Cholesterol** | Numerical | Serum cholesterol (mm/dl). Note: 0 values represent missing data |
| **FastingBS** | Binary | Fasting blood sugar (1 if > 120 mg/dl, 0 otherwise) |
| **RestingECG** | Categorical | Resting ECG results (Normal, ST, LVH) |
| **MaxHR** | Numerical | Maximum heart rate recorded |
| **ExerciseAngina** | Binary | Exercise-induced angina (Y: Yes, N: No) |
| **Oldpeak** | Numerical | ST depression induced by exercise relative to rest |
| **ST_Slope** | Categorical | Slope of peak exercise ST segment (Up, Flat, Down) |
| **HeartDisease** | Target | Output class (1: Disease, 0: Healthy) |

## 🔧 Implementation Pipeline

### Phase 1: Data Preprocessing & Feature Engineering

#### 1. Missing Data Management
- Identified missing values coded as 0 in Cholesterol and RestingBP columns
- **Chosen Approach**: Row deletion
  - **Rationale**: Zero values in Cholesterol indicate measurement errors, not valid data. Removing these records prevents distribution distortion of important features.
  - **Alternative approaches considered**:
    - Imputation (rejected due to high missing count causing bias)
    - Missing flag encoding (not used for simplicity)

#### 2. Data Encoding
- **Sex**: Label Encoding (binary: 0/1)
- **ChestPainType**: Ordinal Encoding (0-3) - inherent ordering exists
- **RestingECG**: Ordinal Encoding (0-2) - severity ordering
- **ExerciseAngina**: Label Encoding (binary: 0/1)
- **ST_Slope**: Ordinal Encoding (0-2) - natural progression

#### 3. Feature Selection & Analysis
- Correlation matrix analysis performed
- Key features identified: ST_Slope, ExerciseAngina, Oldpeak, ChestPainType
- No features removed despite weak correlations due to medical significance

### Phase 2: Model Development & Training

#### 1. Train-Test Split
- **Split ratio**: 80% training, 20% testing
- **Stratification**: Applied to maintain class balance
- **Random state**: Fixed at 42 for reproducibility

#### 2. Normalization & Standardization
- **Method**: StandardScaler (zero mean, unit variance)
- **Applied to**: Gaussian Naive Bayes (assumes normal distribution)
- **Not applied to**: Tree-based models (Decision Tree, Random Forest, XGBoost) - they are scale-invariant

#### 3. Model Implementations

##### Naive Bayes
- **Variant**: Gaussian Naive Bayes
- **Rationale**: Dataset contains continuous numerical features; Gaussian assumption fits the data distribution
- **Key Assumption**: Feature independence given the class label
- **Reality Check**: Features like Age, RestingBP, and Cholesterol are correlated in real medical data, but Naive Bayes still performs reasonably well due to robust probability estimation

##### Decision Tree
- **Initial Model**: No depth constraints → High overfitting (train accuracy >> test accuracy)
- **Optimized Model**: 
  - `max_depth=6`
  - `min_samples_split=10`
  - `min_samples_leaf=5`
- **Result**: Reduced train-test accuracy gap, improved generalization

##### Random Forest
- **Ensemble Method**: Bagging approach
- **Configuration**: 100 estimators
- **Advantage**: Reduces variance and overfitting by averaging multiple decision trees trained on random data subsets
- **Robustness**: More resistant to noise compared to single decision trees

##### XGBoost
- **Ensemble Method**: Boosting approach
- **Key Difference**: Trees built sequentially to correct previous errors (vs. Random Forest's parallel independent trees)
- **Status**: Industry-standard for data science competitions

### Phase 3: Evaluation & Analysis

#### Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|----------|
| **Accuracy** | \((TP + TN) / Total\) | Overall correctness |
| **Precision** | \(TP / (TP + FP)\) | Positive prediction reliability |
| **Recall** | \(TP / (TP + FN)\) | Sensitivity to positive cases |
| **F1-Score** | \(2 \times (Precision \times Recall) / (Precision + Recall)\) | Harmonic mean |

#### Averaging Methods
- **Macro**: Unweighted mean (treats all classes equally)
- **Micro**: Aggregated across all instances
- **Weighted**: Weighted by class support

#### Critical Insight for Medical Diagnosis
**In dangerous disease detection (like heart attacks), Recall is more important than Precision.**

**Why?** Because:
- **False Negatives (FN)** are dangerous: Missing a sick patient could be fatal
- **False Positives (FP)** are less critical: Healthy patients flagged for further testing (safer)

Therefore, models with higher Recall minimize the risk of missing diseased patients.

## 📈 Results

All models achieved strong performance:

- **Decision Tree**: Balanced performance with interpretability
- **Naive Bayes**: Fast training, good baseline
- **Random Forest**: Reduced overfitting, high stability
- **XGBoost**: Best overall performance metrics

**Recommendation**: Based on Recall scores (minimizing False Negatives), the best model for hospital deployment is identified in the analysis output.

## 📊 Visualizations

The project generates comprehensive visualizations including:

### Feature Correlation Matrix
![Correlation Heatmap](images/correlation_heatmap.jpg)

### Feature Importance Comparison
![Feature Importance](images/feature_importance.jpg)

### Model Performance Metrics
![Model Metrics](images/model_metrics.jpg)

### False Negatives Analysis
![False Negatives](images/false_negatives.jpg)

### Decision Tree Visualization
![Decision Tree](images/decision_tree.jpg)

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **scikit-learn** - ML algorithms and evaluation
- **XGBoost** - Gradient boosting
- **matplotlib** & **seaborn** - Visualization

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/VictimPickle/Heart-Disease-ML-Prediction.git
cd Heart-Disease-ML-Prediction

# Install required packages
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## 🚀 Usage

```bash
python heart_disease_analysis.py
```

### Output Files Generated:
- `01_correlation_heatmap.png` - Feature correlation visualization
- `02_feature_importance.png` - Feature importance comparison
- `03_model_metrics.png` - Model performance metrics
- `04_confusion_matrices.png` - Confusion matrices for all models
- `05_false_negatives.png` - False negative comparison
- `06_decision_tree.png` - Decision tree visualization
- `model_comparison_results.csv` - Numerical results
- `processed_heart_data.csv` - Cleaned and encoded dataset

## 🧪 Key Findings

1. **Most Important Features**: ST_Slope, Oldpeak, ExerciseAngina emerged as top predictors
2. **Overfitting Control**: Tree pruning and ensemble methods effectively reduced overfitting
3. **Model Comparison**: Ensemble methods (Random Forest, XGBoost) outperformed single models
4. **Medical Context**: Recall prioritized over precision to minimize missing sick patients

## 📚 Project Structure

```
├── heart_disease_analysis.py    # Main analysis script
├── heart.csv                     # Dataset
├── README.md                     # Project documentation
├── HW7_AI.pdf                    # Original assignment (Persian)
├── images/                       # Visualization outputs
│   ├── correlation_heatmap.jpg
│   ├── feature_importance.jpg
│   ├── model_metrics.jpg
│   ├── false_negatives.jpg
│   └── decision_tree.jpg
└── outputs/                      # Generated CSV results
```

## 👨‍💻 Author

**Mobin Ghorbani**
- Computer Science Student
- University of Tehran (UT)
- GitHub: [@VictimPickle](https://github.com/VictimPickle)

## 📝 License

This project is part of an academic assignment for educational purposes.

## 🙏 Acknowledgments

- Dataset source: Cardiovascular disease dataset
- Course: Artificial Intelligence - University of Tehran
- Faculty of Mathematics, Statistics & Computer Science

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact via email.

---

⭐ If you found this project helpful, please consider giving it a star!
