# Medical Insurance Cost Prediction

## Overview

This project uses Multiple Linear Regression to predict medical insurance charges based on demographic and health-related features.

The main purpose of this project is to practice the complete machine learning workflow including exploratory data analysis, data preprocessing, feature encoding, feature scaling, train-test splitting, regression modeling, and model evaluation.

## Dataset

The dataset contains information about medical insurance customers.

Original features include:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region
- Insurance charges

The target variable is:

**charges** — the medical insurance cost.

## Project Workflow

The project includes:

- Data loading and inspection
- Exploratory Data Analysis (EDA)
- Duplicate removal
- Categorical feature encoding
- BMI category creation
- Feature scaling
- Feature selection
- Train-test split
- Multiple Linear Regression
- Model evaluation using R² and Adjusted R²

## Model

The model used in this project is:

**Multiple Linear Regression**

The dataset was divided into training and testing sets before evaluating the regression model.

## Results

The model achieved approximately:

- **R² Score:** 0.772
- **Adjusted R² Score:** 0.769

An R² of approximately 0.77 means that the regression model explains about 77% of the variation in insurance charges in the test data.

The Adjusted R² is close to the R² score, suggesting that the selected predictors contribute meaningful information without a large penalty for unnecessary features.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Purpose

This is a learning-focused machine learning project created to practice regression, preprocessing, feature engineering, and model evaluation.

> Note: This project is for educational purposes only and should not be used for real insurance pricing or medical decision-making.
