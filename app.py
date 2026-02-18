import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("models/trained_model.pkl")
df = pd.read_csv("data/Training.csv")

st.set_page_config(page_title="AI Disease Predictor", layout="wide")

st.title("🩺 AI-Based Disease Prediction System")
st.markdown("Predict disease using Machine Learning with explainable insights.")

# Sidebar symptom selection
st.sidebar.header("Select Symptoms")

symptoms = df.columns[:-1]
selected = st.sidebar.multiselect("Choose Symptoms", symptoms)

input_data = pd.DataFrame([[1 if col in selected else 0 for col in symptoms]], columns=symptoms)

if st.sidebar.button("Predict Disease"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader(f"🧠 Predicted Disease: **{prediction}**")

    # Probability Visualization
    prob_df = pd.DataFrame({
        "Disease": model.classes_,
        "Probability": probabilities
    }).sort_values(by="Probability", ascending=False)

    fig = px.bar(
        prob_df.head(5),
        x="Probability",
        y="Disease",
        orientation="h",
        title="Top 5 Disease Probabilities"
    )

    st.plotly_chart(fig, use_container_width=True)

# Feature Importance
st.subheader("📊 Symptom Importance")

importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Symptom": symptoms,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(20)

fig2 = px.bar(
    feat_df,
    x="Importance",
    y="Symptom",
    orientation="h",
    title="Top 20 Important Symptoms"
)

st.plotly_chart(fig2, use_container_width=True)
