import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

# Data Loading
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(r"C:\Users\Varshini V\OneDrive\Documents\VS_code\venv\Content_Monetization\cleaned_dataset.csv")

# Encoding Maps
CATEGORY_ENCODING = {"Education": 0, "Entertainment": 1, "Gaming": 2, "Lifestyle": 3, "Music": 4, "Tech": 5}
DEVICE_ENCODING = {"TV": 0, "Tablet": 1, "Mobile": 2, "Desktop": 3}
COUNTRY_ENCODING = {"AU": 0, "CA": 1, "DE": 2, "IN": 3, "UK": 4, "US": 5}

df["category"] = df["category"].map(CATEGORY_ENCODING)
df["device"] = df["device"].map(DEVICE_ENCODING)
df["country"] = df["country"].map(COUNTRY_ENCODING)

 
# Features & Target
X = df.drop(columns=["ad_revenue_usd"])
y = df["ad_revenue_usd"]
FEATURES = X.columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model Registry
MODELS = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Lasso Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.01))
    ]),
    "Random Forest": Pipeline([
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("model", DecisionTreeRegressor(random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("model", GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
}

def get_user_input(df):
    return {
        "views": st.sidebar.number_input("Views", 0,  value=int(df["views"].median())),
        "likes": st.sidebar.number_input("Likes", 0,  value=int(df["likes"].median())),
        "comments": st.sidebar.number_input("Comments", 0,  value=int(df["comments"].median())),
        "watch_time_minutes": st.sidebar.number_input("Watch Time (minutes)", 0.0,  value=float(df["watch_time_minutes"].median())),
        "video_length_minutes": st.sidebar.number_input("Video Length (minutes)", 0.1,  value=float(df["video_length_minutes"].median())),
        "subscribers": st.sidebar.number_input("Subscribers", 0,  value=int(df["subscribers"].median())),
        "Year": st.sidebar.number_input("Year", 2015, 2035, 2024),
        "Month": st.sidebar.number_input("Month", 1, 12, 6),
        "Day": st.sidebar.number_input("Day", 1, 31, 15),
        "engagement rate": st.sidebar.number_input("Engagement Rate", 0.0,  value=float(df["engagement rate"].median())),
        "category": CATEGORY_ENCODING[st.sidebar.selectbox("Category", list(CATEGORY_ENCODING.keys()))],
        "device": DEVICE_ENCODING[st.sidebar.selectbox("Device", list(DEVICE_ENCODING.keys()))],
        "country": COUNTRY_ENCODING[st.sidebar.selectbox("Country", list(COUNTRY_ENCODING.keys()))]
    }

# Page Configuration
st.set_page_config(page_title="Ad Revenue Predictor", layout="wide")
st.title("📈 Ad Revenue Prediction")

# Sidebar
st.sidebar.header("⚙️ Model Selection")
selected_model_name = st.sidebar.selectbox("Choose Algorithm", list(MODELS.keys()))
selected_model = MODELS[selected_model_name]
selected_model.fit(x_train, y_train)
 
st.sidebar.header("🎯 Enter Video Details")
user_input = get_user_input(df)

# Prediction
def predict_revenue(model, user_input):
    input_df = pd.DataFrame([user_input])[FEATURES]
    return model.predict(input_df)[0]

def evaluate_model(model, x_test, y_test):
    preds = model.predict(x_test)
    return {
        "R² Score": r2_score(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "Predictions": preds
    }

if st.sidebar.button("🚀 Predict Revenue and Performance"):
    prediction = predict_revenue(selected_model, user_input)
    metrics = evaluate_model(selected_model, x_test, y_test)

    st.subheader("💰 Predicted Ad Revenue")
    st.metric("Estimated Revenue (USD)", f"${prediction:,.2f}")
    st.caption(f"Model Used: **{selected_model_name}**")

    st.subheader(f"📊 Performance — {selected_model_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{metrics['R² Score']:.3f}")
    c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    c3.metric("MAE", f"{metrics['MAE']:.2f}")

 
# Overall Model Comparison
 
if st.sidebar.button("📈 Overall Model Comparison"):
    rows = []
    for name, mdl in MODELS.items():
        mdl.fit(x_train, y_train)
        metrics = evaluate_model(mdl, x_test, y_test)
        rows.append({"Model": name, "R² Score": metrics["R² Score"], "RMSE": metrics["RMSE"], 
                     "MAE": metrics["MAE"], "Predicted Revenue": np.mean(metrics["Predictions"])})

    perf_df = pd.DataFrame(rows)
    st.subheader("📈 Overall Model Performance")
    st.dataframe(perf_df.style.format({"R² Score": "{:.3f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}), use_container_width=True)

    st.subheader("📊 Performance Comparison Chart")
    fig = px.line(perf_df, x="Model", y="R² Score", title="R² Scores of Different Models")
    st.plotly_chart(fig)