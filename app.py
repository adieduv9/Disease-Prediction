import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Disease Predictor", layout="wide")

# Load dataset
df = pd.read_csv("data/Training.csv")

# Load model
model = joblib.load("models/trained_model.pkl")

# Prepare data
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Train-test split (for confusion matrix display only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

y_pred = model.predict(X_test)

# ===============================
# UI START
# ===============================

st.title("🩺 AI-Based Disease Prediction System")
st.markdown("### Predict disease using Machine Learning with explainable insights")

st.sidebar.header("🧾 Select Symptoms")

symptoms = X.columns.tolist()

selected_symptoms = st.sidebar.multiselect(
    "Choose symptoms:",
    symptoms
)

input_data = pd.DataFrame(
    [[1 if symptom in selected_symptoms else 0 for symptom in symptoms]],
    columns=symptoms
)

if st.sidebar.button("Predict Disease"):

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader(f"🧠 Predicted Disease: **{prediction}**")

    # Probability Chart
    prob_df = pd.DataFrame({
        "Disease": model.classes_,
        "Probability": probabilities
    }).sort_values(by="Probability", ascending=False)

    fig = px.bar(
        prob_df.head(5),
        x="Probability",
        y="Disease",
        orientation="h",
        title="Top 5 Disease Probabilities",
        color="Probability",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# Feature Importance
# ===============================

st.markdown("---")
st.subheader("📊 Top 20 Important Symptoms")

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
    title="Feature Importance",
    color="Importance",
    color_continuous_scale="Viridis"
)

st.plotly_chart(fig2, use_container_width=True)

# ===============================
# Confusion Matrix
# ===============================

st.markdown("---")
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(plt)
