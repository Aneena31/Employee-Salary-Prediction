
import streamlit as st
import pandas as pd
import joblib

# Load the trained model (should include scaler + classifier)
model = joblib.load("best_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")  # Trained on 'income'

st.set_page_config(page_title="Employee Salary Classification", layout="wide")
st.title("💰 Employee Salary Classification")

st.markdown("""
Predicts whether an employee earns **>50K** or **≤50K** based on demographic and work-related features.
""")

# Sidebar input
st.sidebar.header("Input Employee Details")

# --- HARDCODED OPTIONS (from your LabelEncoder training) ---
workclass_options = {
    "Federal-gov": 0, "Local-gov": 1, "Others": 2, "Private": 3,
    "Self-emp-inc": 4, "Self-emp-not-inc": 5, "State-gov": 6
}

marital_status_options = {
    "Divorced": 0, "Married-spouse-absent": 1, "Married-civ-spouse": 2,
    "Never-married": 4, "Separated": 5, "Widowed": 6
}

occupation_options = {
    "Adm-clerical": 0, "Armed-Forces": 1, "Farming-fishing": 2, "Exec-managerial": 3,
    "Craft-repair": 4, "Handlers-cleaners": 5, "Machine-op-inspct": 6,
    "Prof-specialty": 6, "Other-service": 8, "Sales": 9, "Priv-house-serv": 10,
    "Protective-serv": 11, "Transport-moving": 12, "Tech-support": 13, "Others": 8
}

relationship_options = {
    "Husband": 0, "Wife": 1, "Other-relative": 2,
    "Not-in-family": 3, "Own-child": 4, "Unmarried": 5
}

race_options = {
    "Amer-Indian-Eskimo": 0, "Asian-Pac-Islander": 1,
    "Black": 2, "Other": 3, "White": 4
}

gender_options = {"Female": 0, "Male": 1}

native_country_options = {
    'Cambodia': 0, 'Canada': 1, 'China': 2, 'Columbia': 3, 'Cuba': 4, 'Dominican-Republic': 5,
    'Ecuador': 6, 'El-Salvador': 7, 'England': 8, 'France': 9, 'Germany': 10, 'Greece': 11,
    'Guatemala': 12, 'Haiti': 13, 'Holand-Netherlands': 14, 'Honduras': 15, 'Hong': 16,
    'Hungary': 17, 'India': 18, 'Iran': 19, 'Ireland': 20, 'Italy': 21, 'Jamaica': 22,
    'Japan': 23, 'Laos': 24, 'Mexico': 25, 'Nicaragua': 26, 'Outlying-US(Guam-USVI-etc)': 27,
    'Peru': 28, 'Philippines': 29, 'Poland': 30, 'Portugal': 31, 'Puerto-Rico': 32,
    'Scotland': 33, 'South': 34, 'Taiwan': 35, 'Thailand': 36, 'Trinadad&Tobago': 37,
    'United-States': 38, 'Vietnam': 39, 'Yugoslavia': 40, 'Others': 41
}

# --- Form Inputs ---
age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", list(workclass_options.keys()))
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 10000, 1000000, 250000, step=1000)
education_num = st.sidebar.slider("Education Number", 5, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", list(marital_status_options.keys()))
occupation = st.sidebar.selectbox("Occupation", list(occupation_options.keys()))
relationship = st.sidebar.selectbox("Relationship", list(relationship_options.keys()))
race = st.sidebar.selectbox("Race", list(race_options.keys()))
gender = st.sidebar.radio("Gender", list(gender_options.keys()))
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 10000, 0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", list(native_country_options.keys()))

# --- Build input DataFrame ---
input_data = pd.DataFrame([{
    'age': age,
    'workclass': workclass_options[workclass],
    'fnlwgt': fnlwgt,
    'educational-num': education_num,
    'marital-status': marital_status_options[marital_status],
    'occupation': occupation_options[occupation],
    'relationship': relationship_options[relationship],
    'race': race_options[race],
    'gender': gender_options[gender],
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country_options[native_country]
}])

st.subheader("🔍 Input Summary")
st.dataframe(input_data)

# --- Predict ---
if st.button("Predict Salary Class"):
    prediction = model.predict(input_data)[0]
    pred_label = label_encoder.inverse_transform([prediction])[0]
    st.success(f"🎯 Predicted Income Class: **{pred_label}**")

# --- Batch Prediction ---
st.markdown("---")
st.markdown("### 📂 Batch Prediction (CSV)")

uploaded_file = st.file_uploader("Upload a CSV file with appropriate input features", type=["csv"])

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("🔎 Preview of uploaded data:")
    st.write(batch_data.head())

    predictions = model.predict(batch_data)
    batch_data['Predicted Income'] = label_encoder.inverse_transform(predictions)

    st.success("✅ Batch prediction completed!")
    st.write(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "batch_predictions.csv", "text/csv")
