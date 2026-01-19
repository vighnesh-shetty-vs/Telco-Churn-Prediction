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

## 🔍 Model Interpretability (Explainable AI)
A key feature of this dashboard is the **Dynamic Feature Importance** chart. Unlike static models, this dashboard provides a local explanation for every single prediction. 

* **Global Insight:** The model identifies that long-term contracts and low monthly charges generally reduce churn.
* **Local Insight:** For an individual customer, the app calculates exactly how much their specific attributes (e.g., high Fiber Optic costs) contributed to their risk score.

> **Business Value:** This allows customer success teams to see not just *who* might leave, but *why*, enabling personalized retention strategies (e.g., offering a contract upgrade if 'Month-to-month' is the primary churn driver).

## 📈 Metrics & Business Impact
In this project, we prioritize **Recall** over Accuracy. 

* **Recall:** Our ability to find all customers who actually churn.
* **Business Logic:** The cost of a "False Negative" (failing to predict a churner) is the loss of total future revenue. The cost of a "False Positive" (giving a discount to a loyal customer) is a minor margin hit.

By using **SMOTE**, we improved the model's recall on the minority class, ensuring the business captures the maximum number of at-risk customers.

![Sample 1 - High Risk Customer](assets/Sample_High_Risk_customer_1.png)
![Sample 1 - High Risk Customer](assets/Sample_High_Risk_customer_2.png)

![Sample 2 - Low Risk Customer](assets/Sample_Low_Risk_customer_1.png)
![Sample 1 - Low Risk Customer](assets/Sample_Low_Risk_customer_2.png)

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
