import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load trained model and scaler
# -----------------------------
model = joblib.load("model/gaming_model.pkl")
scaler = joblib.load("model/scaler.pkl")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Gaming Performance Predictor",
    page_icon="🎮",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------
st.title("🎮 AI Gaming Performance Predictor")

st.write(
    "Enter a player's gaming statistics and the trained "
    "Machine Learning model will predict the performance category."
)

st.info(
    "This project uses a Logistic Regression model trained "
    "on a synthetically generated gaming dataset."
)


# -----------------------------
# User Inputs
# -----------------------------
st.header("📊 Player Statistics")

matches_played = st.number_input(
    "Matches Played",
    min_value=1,
    max_value=10000,
    value=100
)

win_rate = st.slider(
    "Win Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

avg_kills = st.number_input(
    "Average Kills",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

avg_deaths = st.number_input(
    "Average Deaths",
    min_value=0.0,
    max_value=100.0,
    value=8.0
)

avg_assists = st.number_input(
    "Average Assists",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

accuracy = st.slider(
    "Accuracy (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

headshot_rate = st.slider(
    "Headshot Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

avg_damage = st.number_input(
    "Average Damage",
    min_value=0.0,
    max_value=1000.0,
    value=300.0
)

playtime_hours = st.number_input(
    "Playtime (Hours)",
    min_value=0.0,
    max_value=10000.0,
    value=500.0
)

recent_win_rate = st.slider(
    "Recent Win Rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)


# -----------------------------
# Create input DataFrame
# -----------------------------
input_data = pd.DataFrame({
    "matches_played": [matches_played],
    "win_rate": [win_rate],
    "avg_kills": [avg_kills],
    "avg_deaths": [avg_deaths],
    "avg_assists": [avg_assists],
    "accuracy": [accuracy],
    "headshot_rate": [headshot_rate],
    "avg_damage": [avg_damage],
    "playtime_hours": [playtime_hours],
    "recent_win_rate": [recent_win_rate]
})


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Performance"):

    # Scale the input using the same scaler
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(input_scaled)[0]

    classes = model.classes_

    probability_df = pd.DataFrame({
        "Performance": classes,
        "Probability": probabilities
    })

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    ).round(2)


    # -----------------------------
    # Display prediction
    # -----------------------------
    st.subheader("🎯 Prediction")

    if prediction == "HIGH":
        st.success("🔥 Predicted Performance: HIGH")

    elif prediction == "MEDIUM":
        st.warning("⚡ Predicted Performance: MEDIUM")

    else:
        st.error("📉 Predicted Performance: LOW")


    # -----------------------------
    # Probability
    # -----------------------------
    st.subheader("📈 Prediction Probability")

    st.dataframe(
        probability_df,
        use_container_width=True
    )

    st.bar_chart(
        probability_df.set_index("Performance")
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "AI-Based Gaming Performance Prediction System "
    "using Machine Learning"
)