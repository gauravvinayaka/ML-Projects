# Heart Disease Prediction ❤️

A dark-themed Streamlit app that classifies the `HeartDisease` label using patient health and exercise-related inputs.

## Features
- 11 user inputs arranged in two columns (6 personal/clinical and 5 exercise/ECG)
- Logistic regression classifier with saved StandardScaler and one-hot feature mapping
- Prediction result card and model score for the heart disease class

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
Select repository `gauravvinayaka/ML-Projects`, branch `main`, and **main file path**:
```
Heart Disease Prediction/app.py
```
Choose a compatible Python version (e.g. Python 3.13) and deploy.

The saved artifacts were produced with scikit-learn 1.7.2.

## Disclaimer
This is a machine-learning portfolio application, not a medically validated diagnostic or emergency decision system. Model scores should not be interpreted as clinical risk.
