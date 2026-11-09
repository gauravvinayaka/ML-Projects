
# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting the presence of heart disease using Machine Learning classification algorithms.

I performed Exploratory Data Analysis (EDA), data preprocessing, feature engineering, model training, and evaluation using five different classification algorithms.

After comparing their performance, **Logistic Regression achieved the highest accuracy of 86.41%**, along with the best F1-score.

Therefore, I selected **Logistic Regression** as the final model and developed an interactive web application using Streamlit.

---

## 📊 Dataset

The project uses the Heart Disease dataset containing:

- **918 patient records**
- **11 input features**
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

---

## 🔍 Exploratory Data Analysis (EDA)

Performed Exploratory Data Analysis to understand the dataset and identify relationships between patient health features and heart disease.

The analysis included:

- Dataset shape and statistical summary
- Data types and missing-value checks
- Class distribution analysis
- Histograms and distribution plots
- Count plots for categorical features
- Box plots and violin plots
- Correlation heatmap

Libraries used for visualization:

- Matplotlib
- Seaborn

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Checked the dataset for missing values
- Applied one-hot encoding to categorical variables
- Preserved numerical feature values
- Split the dataset into training and testing sets (80:20)
- Replaced zero values in Cholesterol and RestingBP using training-data means
- Applied StandardScaler for feature scaling
- Used `fit_transform()` on training data
- Used `transform()` on testing data

After encoding, the model used **15 input feature columns**.

---

## 🧠 Feature Engineering and Analysis

Performed feature analysis using:

### Pearson Correlation

Calculated Pearson correlation coefficients to understand relationships between features and the HeartDisease target.

### Chi-Square Test

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

## 🤖 Machine Learning Models Implemented

I trained and evaluated **five different classification algorithms**:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Machine (SVM)

All five models were evaluated using the same training and testing split.

---

## 📈 Model Performance Comparison

The models were evaluated using **Accuracy** and **F1-score**.

| Classification Model | Accuracy | F1-Score |
|----------------------|----------|----------|
| **Logistic Regression** | **86.41%** | **0.8792** |
| K-Nearest Neighbors (KNN) | 85.33% | 0.8708 |
| Gaussian Naive Bayes | 84.78% | 0.8614 |
| Decision Tree Classifier | 79.89% | 0.8141 |
| Support Vector Machine (SVM) | 85.87% | 0.8762 |

### 🏆 Best Model: Logistic Regression

Based on the model comparison, **Logistic Regression performed the best**, achieving:

- **Accuracy: 86.41%**
- **F1-Score: 0.8792**

It achieved the highest accuracy and F1-score among all five classification algorithms tested.

Therefore, I selected Logistic Regression as the final model for the Heart Disease Prediction web application.

---

## 💾 Model Saving

The trained model and preprocessing components were saved using Joblib.

Files used in the Streamlit application:

- `LogisticRegression_heart.pkl` — Trained Logistic Regression model
- `scaler.pkl` — Fitted StandardScaler
- `columns.pkl` — Feature names and their order

These files allow the Streamlit application to load the trained model and generate predictions using new input data.

---

## 🌐 Streamlit Web Application

Developed an interactive Heart Disease Prediction web application using Streamlit.

### Application Features

- Modern dark-themed interface
- Interactive sliders and dropdown menus
- User-friendly patient health input form
- Two-column layout
- Heart disease classification result
- Model-estimated probability display
- Integration with the saved Machine Learning model

The application takes 11 user-provided health measurements, applies the required feature encoding and scaling, and generates a prediction using the trained Logistic Regression model.

### 🔗 Streamlit Application

[Heart Disease Prediction App](https://mlheart-disease-prediction.streamlit.app/)

---

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

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Exploratory Data Analysis
- Data Cleaning and Preprocessing
- Categorical Feature Encoding
- Feature Engineering and Statistical Analysis
- Training Multiple Classification Models
- Comparing Model Performance
- Evaluating Accuracy and F1-score
- Selecting a Final Machine Learning Model
- Saving Trained Models Using Joblib
- Building Interactive Applications Using Streamlit
- Preparing Machine Learning Projects for Deployment

---

## ⚠️ Disclaimer

This project is intended for Machine Learning learning and portfolio demonstration purposes.

The predictions are not medically validated and should not be used as a substitute for professional medical diagnosis or advice.

---

## 👨‍💻 Developed By

**Gaurav Vinayaka**

Built as part of my learning journey in **Python, Machine Learning, Data Science, and Artificial Intelligence**.
