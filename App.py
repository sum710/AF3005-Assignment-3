import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Financial ML App", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stButton>button {
        background-color: #005555;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover { background-color: #004444; }
</style>
""", unsafe_allow_html=True)

# Title and intro
st.title("📊 Financial ML Dashboard")
st.markdown("Welcome! Analyze stock data and apply machine learning techniques interactively.")
st.image("https://media.giphy.com/media/l0MYt5jPRARuuDelS/giphy.gif", width=300)

# Sidebar input
st.sidebar.header("Data Input")
data_source = st.sidebar.radio("Choose Data Source", ["Upload CSV (Kaggle)", "Yahoo Finance"])

# Session state initialization
for key in ['data', 'model', 'X_train', 'X_test', 'y_train', 'y_test', 'scaler']:
    if key not in st.session_state:
        st.session_state[key] = None

# Load data
if data_source == "Upload CSV (Kaggle)":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.success("Dataset loaded successfully.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
else:
    ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    if st.sidebar.button("Fetch Data"):
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                st.error("No data found for the given ticker and date.")
            else:
                df.dropna(inplace=True)
                st.session_state.data = df
                st.success(f"Data for {ticker} loaded.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# Preview
if st.session_state.data is not None:
    st.subheader("Data Preview")
    st.write(st.session_state.data.head())

# Step 1: Preprocess
if st.button("Step 1: Preprocess Data") and st.session_state.data is not None:
    try:
        numeric = st.session_state.data.select_dtypes(include=[np.number])
        st.session_state.data[numeric.columns] = numeric.fillna(numeric.mean())
        z = np.abs((numeric - numeric.mean()) / numeric.std())
        st.session_state.data = st.session_state.data[(z < 3).all(axis=1)]
        st.success("Preprocessing completed.")
    except Exception as e:
        st.error(f"Preprocessing error: {e}")

# Step 2: Feature engineering
if st.button("Step 2: Feature Engineering") and st.session_state.data is not None:
    try:
        if 'Close' in st.session_state.data.columns:
            df = st.session_state.data
            df['SMA_10'] = df['Close'].rolling(10).mean()
            df['SMA_30'] = df['Close'].rolling(30).mean()
            df.dropna(inplace=True)
            st.session_state.data = df
            features = ['Open', 'High', 'Low', 'Volume']
            target = 'Close'
            corr = [df[f].corr(df[target]) for f in features]
            fig = px.bar(x=features, y=corr, title="Correlation with Close Price")
            st.plotly_chart(fig)
            st.success("Feature engineering completed.")
    except Exception as e:
        st.error(f"Feature engineering error: {e}")

# Step 3: Train/test split
if st.button("Step 3: Train/Test Split") and st.session_state.data is not None:
    try:
        features = ['Open', 'High', 'Low', 'Volume']
        X = st.session_state.data[features]
        y = st.session_state.data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test
        st.plotly_chart(go.Figure(go.Pie(labels=["Train", "Test"],
                                         values=[len(X_train), len(X_test)],
                                         title="Train/Test Split")))
        st.success("Data split into train/test sets.")
    except Exception as e:
        st.error(f"Train/test split error: {e}")

# Step 4: Train model
model_choice = st.selectbox("Choose Model", ["Linear Regression", "Logistic Regression"])
if st.button("Step 4: Train Model") and st.session_state.X_train is not None:
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(st.session_state.X_train)
        X_test_scaled = scaler.transform(st.session_state.X_test)
        st.session_state.scaler = scaler

        if model_choice == "Linear Regression":
            model = LinearRegression()
            y_train = st.session_state.y_train
        else:
            median = st.session_state.y_train.median()
            y_train = (st.session_state.y_train > median).astype(int)
            st.session_state.y_test = (st.session_state.y_test > median).astype(int)
            model = LogisticRegression()

        model.fit(X_train_scaled, y_train)
        st.session_state.model = model
        st.success(f"{model_choice} trained successfully.")
    except Exception as e:
        st.error(f"Model training error: {e}")

# Step 5: Evaluate
if st.button("Step 5: Evaluate Model") and st.session_state.model is not None:
    try:
        X_test_scaled = st.session_state.scaler.transform(st.session_state.X_test)
        y_pred = st.session_state.model.predict(X_test_scaled)

        if model_choice == "Linear Regression":
            rmse = np.sqrt(mean_squared_error(st.session_state.y_test, y_pred))
            st.metric("RMSE", f"{rmse:.2f}")
        else:
            acc = accuracy_score(st.session_state.y_test, y_pred)
            st.metric("Accuracy", f"{acc:.2%}")

        fig = px.scatter(x=st.session_state.y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'},
                         title="Actual vs Predicted")
        st.plotly_chart(fig)
        st.success("Evaluation complete.")
    except Exception as e:
        st.error(f"Evaluation error: {e}")

# Step 6: Visualize & download
if st.button("Step 6: Visualize Results") and st.session_state.model is not None:
    try:
        if model_choice == "Linear Regression":
            importance = pd.DataFrame({
                'Feature': st.session_state.X_train.columns,
                'Coefficient': np.abs(st.session_state.model.coef_)
            })
            fig = px.bar(importance.sort_values(by='Coefficient', ascending=False),
                         x='Feature', y='Coefficient', title="Feature Importance")
            st.plotly_chart(fig)

        results = pd.DataFrame({
            'Actual': st.session_state.y_test,
            'Predicted': st.session_state.model.predict(
                st.session_state.scaler.transform(st.session_state.X_test))
        })
        st.download_button("Download Results", results.to_csv(index=False), "results.csv", "text/csv")
        st.image("https://media.giphy.com/media/3o7aD2r4kQeQfG3W3W/giphy.gif", width=300)
        st.success("Results generated!")
    except Exception as e:
        st.error(f"Visualization error: {e}")

# How to run
st.markdown("""
### 🚀 How to Run Locally
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/AF3005-Assignment-3.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
""")
