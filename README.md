📊 Telco Customer Churn AI Predictor

Live App: https://vighnesh-shetty-vs-telco-churn-prediction-app-x2xli5.streamlit.app/

This project implements an end-to-end Machine Learning solution to predict customer churn for a telecommunications company. It uses Logistic Regression optimized with SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance, providing a functional Streamlit dashboard for real-time predictions.

🎯 Business Case
In the telecom industry, acquiring a new customer is 5–25 times more expensive than retaining an existing one.

Objective: Identify high-risk customers before they leave.

Key Metric: Recall (capturing as many actual churners as possible) and Churn Probability.

Actionable Insight: The business can use these predictions to offer targeted discounts or loyalty programs to "At-Risk" individuals.

🛠️ Technical Stack
Python 3.10+

Machine Learning: Scikit-Learn (Logistic Regression)

Imbalance Handling: Imbalanced-Learn (SMOTE)

Web Framework: Streamlit

Data Manipulation: Pandas & NumPy

🚀 Features
Automated Pipeline: Data cleaning and encoding are handled automatically upon startup.

Class Imbalance Correction: Uses SMOTE to synthesize new examples of the minority class (churners), significantly improving the model's ability to detect churn.

Interactive UI: A sidebar-driven interface allows users to input customer attributes and see immediate risk scores.

💻 How to Run Locally
1. Clone the Repository
Bash

git clone https://github.com/YOUR_USERNAME/Telco-Churn-Prediction.git
cd Telco-Churn-Prediction
2. Set Up Environment
PowerShell

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Launch the App
PowerShell

streamlit run app.py
📈 Model Performance
Because business data is often imbalanced, we prioritized Recall over simple Accuracy.

Without SMOTE: The model often ignores churners because they are the minority.

With SMOTE: The model "learns" the patterns of churners more effectively, leading to a higher detection rate of at-risk customers.

📁 Project Structure
Plaintext

├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Kaggle Dataset
├── app.py                                   # Main Streamlit Application
├── requirements.txt                         # Dependencies
└── README.md                                # Project Documentation
