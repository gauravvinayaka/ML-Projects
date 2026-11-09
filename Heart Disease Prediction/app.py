import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_files():
    folder = Path(__file__).resolve().parent
    model = joblib.load(folder / "LogisticRegression_heart.pkl")
    scaler = joblib.load(folder / "scaler.pkl")
    expected_columns = joblib.load(folder / "columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_files()

# ---------------- DARK THEME ----------------

st.html("""
<style>
.stApp {
    background: #080d18 !important;
    color: #f1f5f9;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #1c203c 0%, #101626 40%, #080d18 90%) !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}
.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #241631, #201b3c 55%, #37203b);
    border: 1px solid #50314e;
    border-radius: 24px;
    padding: 42px;
    margin-bottom: 32px;
    box-shadow: 0 14px 45px rgba(0, 0, 0, 0.3);
}
.hero-label {
    color: #f39ab2;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 15px;
}
.hero h1 {
    color: white;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800;
    margin-bottom: 14px;
    line-height: 1.2;
}
.hero p {
    color: #cbd5e1;
    font-size: 16px;
    line-height: 1.7;
    margin-bottom: 0;
}
.hero-author {
    color: #f5a8bc;
    font-size: 14px;
    font-weight: 600;
    margin-top: 22px;
}
.section-heading {
    color: #f8fafc;
    font-size: 24px;
    font-weight: 750;
    margin-bottom: 8px;
}
.section-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 24px;
}
.form-heading {
    color: #f8fafc;
    font-size: 19px;
    font-weight: 700;
    padding-bottom: 14px;
    margin-bottom: 22px;
    border-bottom: 1px solid #334155;
}
div[data-testid="stForm"] {
    background: #121b2d !important;
    border: 1px solid #303e55 !important;
    border-radius: 22px !important;
    padding: 34px !important;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.25);
}
[data-testid="stWidgetLabel"] p {
    color: #e2e8f0 !important;
    font-weight: 500;
}
div[data-baseweb="select"] > div {
    background-color: #1c2940 !important;
    border: 1px solid #3c4a63 !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
}
div[data-baseweb="select"] span {
    color: #f8fafc !important;
}
[data-baseweb="popover"] [role="listbox"] {
    background-color: #1c2940 !important;
    color: #f8fafc !important;
}
[data-testid="stSlider"] {
    color: #e2e8f0 !important;
}
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #e34a79, #f0648d) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 54px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(90deg, #c93667, #df477a) !important;
    color: white !important;
    box-shadow: 0 6px 22px rgba(227, 74, 121, 0.25);
}
.result-card {
    background: #151f33;
    border: 1px solid #354158;
    border-radius: 18px;
    padding: 28px;
    min-height: 155px;
}
.result-positive { border-left: 4px solid #f87171; }
.result-negative { border-left: 4px solid #34d399; }
.result-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 14px;
}
.result-title {
    color: #f8fafc;
    font-size: 23px;
    font-weight: 750;
    margin-bottom: 12px;
}
.result-description {
    color: #aebbd0;
    font-size: 14px;
    line-height: 1.6;
}
[data-testid="stMetric"] {
    background: #151f33;
    border: 1px solid #354158;
    border-radius: 18px;
    padding: 24px;
}
[data-testid="stMetricLabel"] { color: #aab8cc; }
[data-testid="stMetricValue"] { color: white; }
.footer {
    text-align: center;
    color: #8291aa;
    font-size: 13px;
    padding-top: 32px;
    padding-bottom: 12px;
    line-height: 1.9;
}
</style>
""")

# ---------------- HEADER ----------------

st.html("""
<div class="hero">
    <div class="hero-label">HEART HEALTH ANALYTICS</div>
    <h1>❤️ Heart Disease Prediction</h1>
    <p>
        Explore heart health patterns using your personal,
        clinical, and exercise-related measurements.
        Enter your details below to generate a prediction.
    </p>
    <div class="hero-author">Designed &amp; Developed by Gaurav</div>
</div>
""")

st.html("""
<div class="section-heading">🩺 Health Assessment</div>
<div class="section-subtitle">
    Complete the following fields to generate your prediction.
</div>
""")

# ---------------- INPUT FORM ----------------

with st.form("heart_prediction_form"):
    left, right = st.columns(2, gap="large")

    with left:
        st.html('<div class="form-heading">👤 Personal &amp; Clinical Details</div>')

        age = st.slider("Age (years)", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
        )
        resting_blood_pressure = st.slider(
            "Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120
        )
        cholesterol = st.slider(
            "Cholesterol (mg/dL)", min_value=100, max_value=600, value=200
        )
        fasting_blood_sugar = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"]
        )

    with right:
        st.html('<div class="form-heading">📈 Heart Tests &amp; Exercise</div>')

        resting_ecg = st.selectbox(
            "Resting ECG Results",
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
        )
        max_heart_rate = st.slider(
            "Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150
        )
        exercise_induced_angina = st.selectbox(
            "Exercise-Induced Angina", ["No", "Yes"]
        )
        oldpeak = st.slider(
            "ST Depression (Oldpeak)", min_value=0.0, max_value=10.0,
            value=1.0, step=0.1
        )
        st_slope = st.selectbox(
            "ST Slope", ["Upsloping", "Flat", "Downsloping"]
        )

    st.write("")
    submitted = st.form_submit_button(
        "❤️ Predict Heart Disease",
        use_container_width=True,
        type="primary"
    )

# ---------------- PREDICTION ----------------

if submitted:
    input_data = pd.DataFrame({
        "Age": [age],
        "RestingBP": [resting_blood_pressure],
        "Cholesterol": [cholesterol],
        "MaxHR": [max_heart_rate],
        "Oldpeak": [oldpeak],
        "FastingBS": [1 if fasting_blood_sugar == "Yes" else 0],
        "Sex_M": [1 if sex == "Male" else 0],
        "ChestPainType_ATA": [1 if chest_pain_type == "Atypical Angina" else 0],
        "ChestPainType_NAP": [1 if chest_pain_type == "Non-Anginal Pain" else 0],
        "ChestPainType_TA": [1 if chest_pain_type == "Typical Angina" else 0],
        "RestingECG_Normal": [1 if resting_ecg == "Normal" else 0],
        "RestingECG_ST": [1 if resting_ecg == "ST-T Wave Abnormality" else 0],
        "ExerciseAngina_Y": [1 if exercise_induced_angina == "Yes" else 0],
        "ST_Slope_Flat": [1 if st_slope == "Flat" else 0],
        "ST_Slope_Up": [1 if st_slope == "Upsloping" else 0],
    })

    input_data = input_data[expected_columns]
    scaled_input = scaler.transform(input_data)

    prediction = int(model.predict(scaled_input)[0])
    probability_class_1 = float(
        model.predict_proba(scaled_input)[0][list(model.classes_).index(1)]
    )

    st.write("")
    st.markdown("## ❤️ Your Prediction Result")
    result_col, score_col = st.columns([1.4, 1], gap="large")

    with result_col:
        if prediction == 1:
            st.html("""
            <div class="result-card result-positive">
                <div class="result-label">PREDICTION RESULT</div>
                <div class="result-title">Heart Disease Indicated</div>
                <div class="result-description">
                    The model detected patterns associated with the
                    heart disease class.
                </div>
            </div>
            """)
        else:
            st.html("""
            <div class="result-card result-negative">
                <div class="result-label">PREDICTION RESULT</div>
                <div class="result-title">No Heart Disease Indicated</div>
                <div class="result-description">
                    The model classified these inputs into the
                    no-heart-disease class.
                </div>
            </div>
            """)

    with score_col:
        st.metric(
            "Model Score — Heart Disease Class",
            f"{probability_class_1 * 100:.1f}%"
        )
        st.progress(probability_class_1)
        st.caption("Model estimate, not a clinically validated risk percentage.")

st.html("""
<div class="footer">
    Developed with ❤️ by Gaurav
    <div>Python • Streamlit • Machine Learning</div>
    <div>Not intended for medical diagnosis or emergency decisions.</div>
</div>
""")
