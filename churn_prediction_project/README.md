# 📡 Telco Customer Churn Prediction — ML Pipeline + Streamlit App

A complete end-to-end churn prediction system built using the **Telco Customer Churn Dataset**.  
This project demonstrates industry-level ML workflow: **data cleaning**, **feature engineering**,  
**model training**, **explainable ML**, and a fully functional **Streamlit web app for real-time predictions**.

---

## 🚀 Project Highlights

- **End-to-end ML pipeline** (DataPrep → Model → API → UI)
- Feature engineering includes:
  - `services_count`
  - `is_monthly_contract`
  - `recent_drop`
- Models trained:
  - Logistic Regression  
  - Random Forest  
  - XGBoost (best performing)
- **Interactive Streamlit app** for real-time predictions
- Clean modular code (`src/` folder)
- Fully deployable structure for production & cloud hosting

---

## 📁 Project Structure

churn_prediction_project/
│
├── data/
│ ├── raw/ # Original dataset
│ └── processed/ # Cleaned + engineered files
│
├── models/
│ └── logistic_model.joblib # Saved ML pipeline
│
├── notebooks/
│ └── eda.ipynb # EDA + visualizations
│
├── src/
│ ├── data_prep.py # Cleaning + feature engineering
│ ├── train_model.py # Pipeline training + evaluation
│ └── predict.py # Inference script
│
├── streamlit/
│ └── app.py # Streamlit web app
│
├── reports/
│ └── figures/ # EDA visuals 
│
└── README.md


---

## 📊 Dataset

**Source:**  
IBM Telco Customer Churn Dataset (Kaggle)

**Target:**  
`Churn` → 1 (Yes), 0 (No)

**Key Features Used:**

- `tenure`, `MonthlyCharges`, `TotalCharges`
- `Contract`, `PaymentMethod`, `InternetService`
- Service add-ons (Security, TechSupport, Streaming)
- Engineered features:
  - `services_count`
  - `is_monthly_contract`
  - `recent_drop`

---

## 🛠️ Feature Engineering

### 1. `services_count`
Counts total subscribed services:
- OnlineSecurity  
- DeviceProtection  
- TechSupport  

### 2. `is_monthly_contract`
Binary indicator for high-risk contract type.

### 3. `recent_drop`
If tenure ≤ 3 months → customer likely churning early.

---

## 🤖 Model Development

### Models Trained:
| Model | Recall | ROC-AUC | Notes |
|-------|--------|---------|-------|
| Logistic Regression | 0.76 | 0.81 | Baseline |
| Random Forest | 0.82 | 0.85 | Good stability |
| **XGBoost** | **0.86** | **0.88** | Best performer |

> **Business Metric:** Prioritized **Recall** to catch as many churners as possible.

---

## 🔍 Explainability (SHAP)

### Global SHAP → Top Drivers
- High monthly charges  
- Month-to-month contracts  
- Short tenure  
- No TechSupport / No OnlineSecurity  

### Local SHAP
The app can show **top 3 reasons** why *one specific customer* may churn.

---

## 🌐 Streamlit Web App

The app allows users to:
- Enter customer attributes
- Get churn probability instantly
- See churn risk label (Likely / Not Likely)

## Live Demo
Check out - [Telco Churn Prediction](https://telcochurnpredict.streamlit.app/)
