
# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting the presence of heart disease using Machine Learning classification algorithms.

I performed Exploratory Data Analysis (EDA), statistical feature analysis, data preprocessing, feature engineering, model training, cross-validation, and hyperparameter tuning using five classification algorithms.

After comparing the models, I applied **5-Fold Cross-Validation** and **GridSearchCV** to identify the best-performing model and hyperparameters.

The final selected model was **K-Nearest Neighbors (KNN) with 9 neighbors**, achieving:

- **Test Accuracy: 86.96%**
- **Test F1-Score: 0.8846**
- **Test Recall: 85.98%**
- **Best Cross-Validation Accuracy: 85.83%**

I also developed an interactive web application using **Streamlit**, allowing users to enter health-related measurements and generate predictions.

### 🌐 Live Application

**[❤️ Try the Heart Disease Prediction App](https://mlheart-disease-prediction.streamlit.app/)**

---

## 📊 Dataset

The project uses a Heart Disease dataset containing:

- **918 patient records**
- **11 original input features**
- **1 target variable: HeartDisease**

### Target Variable

- `0` — No Heart Disease
- `1` — Heart Disease

### Input Features

1. Age
2. Sex
3. Chest Pain Type
4. Resting Blood Pressure
5. Cholesterol
6. Fasting Blood Sugar
7. Resting ECG
8. Maximum Heart Rate
9. Exercise-Induced Angina
10. Oldpeak (ST Depression)
11. ST Slope

After one-hot encoding and feature selection, the models used **15 input feature columns**.

---

## 🔍 Exploratory Data Analysis (EDA)

Performed Exploratory Data Analysis to understand the dataset, examine feature distributions, and identify relationships between patient health characteristics and heart disease.

The analysis included:

- Dataset shape and statistical summary
- Data type inspection
- Missing-value checks
- Target class distribution
- Histograms and distribution plots
- Count plots for categorical features
- Box plots and violin plots
- Correlation analysis and heatmaps

### Visualization Libraries

- Matplotlib
- Seaborn

---

## 🧠 Feature Analysis

Performed statistical analysis to understand relationships between input features and the target variable.

### 1. Pearson Correlation

Calculated Pearson correlation coefficients to measure relationships between numerical and encoded features and the `HeartDisease` target.

### 2. Chi-Square Test

Applied the Chi-Square test to analyze associations between categorical variables and heart disease.

Categorical features analyzed included:

- Sex
- ChestPainType
- RestingECG
- ExerciseAngina
- ST_Slope
- FastingBS

These analyses helped me understand feature relationships and practice statistical techniques used in Machine Learning.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Checked for missing and invalid values
- Applied one-hot encoding to categorical variables
- Selected 15 encoded input features
- Split the dataset into training and testing sets (80:20)
- Used `SimpleImputer` to replace zero values in `Cholesterol` and `RestingBP` with mean values
- Applied `StandardScaler` for feature scaling
- Used Scikit-learn `Pipeline` and `ColumnTransformer` for preprocessing and model training

### Preprocessing Pipeline

The final Machine Learning pipeline consists of:

1. **ColumnTransformer** — Applies preprocessing to selected columns
2. **SimpleImputer** — Handles zero values in Cholesterol and RestingBP
3. **StandardScaler** — Standardizes features
4. **KNeighborsClassifier** — Performs classification

By placing preprocessing inside the pipeline, the transformations are fitted using the appropriate training data during cross-validation.

This helps prevent preprocessing-related data leakage between training and validation folds.

---

## 🤖 Machine Learning Models Implemented

I trained and evaluated five classification algorithms:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Machine (SVM)

Each model was evaluated using the same train-test split.

### Initial Model Performance

| Classification Model | Test Accuracy | F1-Score |
|----------------------|---------------|----------|
| Logistic Regression | 86.41% | 0.8792 |
| K-Nearest Neighbors | 85.33% | 0.8708 |
| Gaussian Naive Bayes | 84.78% | 0.8614 |
| Decision Tree Classifier | 79.35% | 0.8137 |
| Support Vector Machine | 85.87% | 0.8762 |

Logistic Regression achieved the highest initial test accuracy and F1-score before hyperparameter tuning.

---

## 🔄 5-Fold Cross-Validation

Implemented **5-Fold Cross-Validation** using Scikit-learn's `cross_val_score()` to evaluate model performance across multiple training and validation splits.

### Cross-Validation Results

| Model | Mean CV Accuracy |
|-------|------------------|
| Logistic Regression | 85.3% |
| KNN | 85.1% |
| Gaussian Naive Bayes | 85.1% |
| Decision Tree | 77.0% |
| SVM | 85.1% |

Cross-validation provided a more comprehensive comparison than relying on a single train-test split.

---

## ⚙️ Hyperparameter Tuning Using GridSearchCV

Implemented **GridSearchCV** to compare multiple classification algorithms and search for their best hyperparameters.

A single Scikit-learn pipeline was used with different classifiers and parameter grids.

### Models Tuned

- Support Vector Machine — `C`, `kernel`
- K-Nearest Neighbors — `n_neighbors`
- Decision Tree — `max_depth`
- Gaussian Naive Bayes — `var_smoothing`
- Logistic Regression — `C`

### GridSearchCV Configuration

- **Cross-validation:** 5 folds
- **Scoring metric:** Accuracy
- **Parallel processing:** `n_jobs=-1`
- **Training data:** Training split only

### Best Model Selected

**K-Nearest Neighbors (KNN)**

- **Optimal number of neighbors:** 9
- **Best CV Accuracy:** 85.83%

GridSearchCV selected KNN with 9 neighbors as the best-performing configuration based on mean cross-validation accuracy.

---

## 🏆 Final Tuned Model Evaluation

The selected KNN pipeline was evaluated on the held-out test dataset.

### Final Performance Metrics

| Metric | Result |
|--------|--------|
| Best Model | KNN |
| Number of Neighbors | 9 |
| Cross-Validation Accuracy | 85.83% |
| Test Accuracy | 86.96% |
| Test F1-Score | 0.8846 |
| Test Recall | 85.98% |

### Confusion Matrix

| Actual / Predicted | No Heart Disease | Heart Disease |
|--------------------|------------------|---------------|
| No Heart Disease | 68 | 9 |
| Heart Disease | 15 | 92 |

### Interpretation

- **68 True Negatives:** Correctly predicted no heart disease
- **92 True Positives:** Correctly predicted heart disease
- **9 False Positives:** Predicted heart disease when the actual class was negative
- **15 False Negatives:** Predicted no heart disease when the actual class was positive

Recall is particularly relevant to this classification problem because it measures the proportion of positive cases correctly identified.

These results are based on the project dataset and should not be interpreted as clinical performance.

---

## 💾 Model Saving and Pipeline Integration

Used **Joblib** to save the trained Machine Learning pipeline and its expected feature columns.

### Saved Files

- `heart_pipeline.pkl` — Complete fitted preprocessing pipeline and tuned KNN model
- `heart_columns.pkl` — Feature column names and their expected order

The saved pipeline automatically performs:

1. Missing-value imputation for selected features
2. Feature scaling
3. KNN classification

This eliminates the need to save or apply a separate scaler during prediction.

---

## 🌐 Streamlit Web Application

Developed an interactive Heart Disease Prediction web application using Streamlit.

### How It Works

1. The user enters 11 health-related measurements.
2. The application encodes categorical inputs into the 15 expected features.
3. Input columns are arranged in the correct training order.
4. The saved pipeline performs preprocessing and scaling.
5. The tuned KNN classifier generates a prediction.
6. The application displays the predicted class and model score.

The displayed score is a model-estimated probability, not a clinically validated disease-risk percentage.

### 🔗 Live Streamlit Application

**[Launch the Heart Disease Prediction App](https://mlheart-disease-prediction.streamlit.app/)**


```

## 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis and Visualization

- NumPy
- Pandas
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- SciPy
- Joblib

### Web Application

- Streamlit

### Development Tools

- Jupyter Notebook
- Visual Studio Code
- GitHub

---

## 🚀 Run the Application Locally

### 1. Clone the Repository

```bash
git clone https://github.com/gauravvinayaka/ML-Projects.git
```

### 2. Navigate to the Repository

```bash
cd ML-Projects
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
python -m streamlit run "Heart Disease Prediction/app.py"
```

The application should open in your browser, usually at:

`http://localhost:8501`

Use a Python environment compatible with the dependencies listed in `requirements.txt`.

---

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Exploratory Data Analysis
- Data Cleaning and Preprocessing
- Categorical Feature Encoding
- Statistical Feature Analysis
- Pearson Correlation and Chi-Square Testing
- Building Scikit-learn Pipelines
- Working with ColumnTransformer and SimpleImputer
- Training Multiple Classification Models
- Evaluating Accuracy, Precision, Recall, and F1-score
- Implementing 5-Fold Cross-Validation
- Performing Hyperparameter Tuning with GridSearchCV
- Comparing Machine Learning Algorithms
- Evaluating Confusion Matrices
- Saving Trained Pipelines with Joblib
- Integrating Machine Learning Models into Streamlit
- Deploying Interactive Applications

---

## ⚠️ Disclaimer

This project is intended for **Machine Learning education, experimentation, and portfolio demonstration purposes only**.

The predictions are not medically validated and should not be used for professional medical diagnosis, treatment, or emergency decisions.

Consult a qualified healthcare professional for medical concerns.

---

## 👨‍💻 Developed By

**Gaurav Vinayaka**

Built as part of my learning journey in **Python, Machine Learning, Data Science, and Artificial Intelligence**.

### 🔗 Project Links

- **[Live Streamlit Application](https://mlheart-disease-prediction.streamlit.app/)**
- **[GitHub Repository](https://github.com/gauravvinayaka/ML-Projects/tree/main/Heart%20Disease%20Prediction)**
