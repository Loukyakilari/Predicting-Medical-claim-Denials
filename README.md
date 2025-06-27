# 🏥 A Machine Learning Framework for Predicting Medical Claim Denial

A Machine Learning Framework for early identification and prevention of insurance claim denials, developed using synthetic healthcare claims data.

## 📌 Project Overview

This project aims to help healthcare providers reduce revenue loss and administrative burden caused by denied claims. Using machine learning, it predicts whether a claim will be approved or denied before submission, and identifies the most likely reason for denial.

- **Data**: 10,000+ synthetic medical claims generated using Faker and custom logic.
- **Tech Stack**: Python, Pandas, Scikit-learn, Flask, HTML/CSS.
- **Models Used**: Logistic Regression, Random Forest, Decision Tree and XGBoost.
- **Deployment**: Flask web application with single and bulk prediction features.

## 📊 Features

- 🧹 **EDA**: Analyze denial trends, claim distribution, denial categories, and provider behavior.
- 🧠 **ML Predictions**: Determine claim approval likelihood and reason for denial (if any).
- 📂 **Predictiion**: Predict for individual or batch claims via UI or CSV upload.
- 🌐 **Flask Web App**: Interactive interface with pages for introduction, EDA, models, and results.


## 📁 Project Structure

```
project-root/
│
├── app/                      # Flask application
│   ├── static/               # CSS and assets
│   ├── templates/            # HTML templates
│   ├── model/                # Trained ML models
│   ├── single_predict.py     # Single prediction logic
│   ├── bulk_predict.py       # Bulk prediction logic
│   └── app.py                # Main Flask server
│
├── data/                     # Sample dataset and generated inputs
│   └── medical_claims.csv
│
├── notebooks/                # EDA and model training notebooks
│   └── claim_modeling.ipynb
│
├── requirements.txt
├── Procfile
├── README.md

```

## 📌 Dataset Description

| Column               | Description                               |
|----------------------|-------------------------------------------|
| `claim_id`           | Unique identifier for each claim          |
| `patient_id`         | Unique identifier for each patient        |
| `specialty`          | Provider specialty (e.g., Cardiology)     |
| `procedure_code`     | CPT/HCPCS code for performed procedure    |
| `diagnosis_code`     | ICD-10 diagnosis code                     |
| `claim_amount`       | Total claim amount billed                 |
| `insurance_provider` | Insurance company                         |
| `is_denied`          | Target variable (1=Denied, 0=Approved)    |
| `denial_reason`      | Reason code or explanation                |

## 🔍 Machine Learning Models

- **Random Forest**
- **Logistic Regression**
- **XGBoost**
- **Decision Tree**

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/loukyakilari/Predicting-Medical-claim-Denials.git
cd Predicting-Medical-claim-Denials
```

### 2. Set up environment

```bash
pip install -r requirements.txt
```

### 3. Launch the web app

```bash
python app.py
```

## 👤 Author

**Loukya Kilari**  
Master’s in Health Informatics – University of Minnesota  
[GitHub](https://github.com/loukyakilari)

