import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# --- STEP 1: DATA PIPELINE ---
@st.cache_data
def load_and_prep_data():
    # Ensure this matches your filename in the 'data' folder
    df = pd.read_csv("data/Telco_Customer_Dataset.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    
    # Target encoding
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Categorical encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded

# --- STEP 2: MODEL TRAINING ---
@st.cache_resource
def train_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle Class Imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_res, y_train_res)
    
    return model, scaler, X.columns

# --- STEP 3: STREAMLIT UI ---
st.set_page_config(page_title="Telco Churn AI", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
This app uses **Logistic Regression** and **SMOTE** to predict if a telecom customer will leave. 
The model is trained on the Kaggle Telco Dataset.
""")

try:
    data = load_and_prep_data()
    model, scaler, feature_columns = train_model(data)

    # SIDEBAR INPUTS
    st.sidebar.header("Customer Information")
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)
    
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    # PROCESS INPUTS
    if st.sidebar.button("Predict Churn Status"):
        # Build the feature vector
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
        input_data['tenure'] = tenure
        input_data['MonthlyCharges'] = monthly
        input_data['TotalCharges'] = total
        
        # Set dummy variables
        if f"Contract_{contract}" in feature_columns: input_data[f"Contract_{contract}"] = 1
        if f"InternetService_{internet}" in feature_columns: input_data[f"InternetService_{internet}"] = 1
        if f"TechSupport_{tech_support}" in feature_columns: input_data[f"TechSupport_{tech_support}"] = 1
        
        # Scale and Predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # DISPLAY RESULTS
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error(f"### High Risk of Churn")
            else:
                st.success(f"### Low Risk (Loyal)")
        
        with col2:
            st.metric("Churn Probability", f"{probability:.2%}")
            st.progress(probability)

        # --- DYNAMIC FEATURE IMPORTANCE ---
        st.divider()
        st.subheader("🔍 What is driving this prediction?")

        # 1. Calculate local contribution: (Coefficient * Scaled Value)
        # This determines how much EACH feature pushed the score UP or DOWN for THIS user
        contributions = model.coef_[0] * input_scaled[0]

        # 2. Create a DataFrame for visualization
        local_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Contribution': contributions
        })

        # 3. Get top 5 drivers (largest positive contributions to churn)
        top_drivers = local_importance.sort_values(by='Contribution', ascending=False).head(5)

        # 4. Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x='Contribution', 
            y='Feature', 
            data=top_drivers, 
            palette='Reds_r', 
            ax=ax
        )
        ax.set_title(f"Top 5 Factors Increasing Risk for this Customer")
        st.pyplot(fig)

        st.write("""
        **Business Insight:** The chart above shows the specific reasons why this customer is considered a risk. 
        Unlike a static global chart, this updates every time you change the sidebar inputs.
        """)

except FileNotFoundError:
    st.error("Dataset not found! Please ensure 'data/Telco_Customer_Dataset.csv' exists.")
