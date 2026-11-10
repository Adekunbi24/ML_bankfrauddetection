# Banking Transaction Intelligence & Fraud Detection Platform

An end-to-end **Data Engineering and Machine Learning** project that analyzes synthetic banking transactions, identifies potentially fraudulent activity, and presents transaction insights through an interactive Streamlit dashboard.

**Live Dashboard:** https://ml-fraud-detection-d.duckdns.org

## Project Overview

This project demonstrates how data engineering, SQL analytics, and machine learning can work together to support transaction monitoring and fraud detection.

The platform uses the PaySim synthetic financial transaction dataset. Transaction data is processed using Python and PostgreSQL, analyzed with SQL, and used to train a Random Forest classification model.

An interactive Streamlit dashboard provides transaction analytics and fraud-risk screening.

## Technology Stack

- **Programming:** Python, SQL
- **Data Engineering:** pandas, PostgreSQL, SQLAlchemy
- **Machine Learning:** scikit-learn, Random Forest, Logistic Regression experiments
- **Dashboard:** Streamlit
- **Deployment:** Docker, AWS EC2, Caddy, HTTPS
- **Version Control:** Git and GitHub

## Data Engineering and Analytics

The project includes Python ETL scripts for loading and processing transaction data in PostgreSQL.

SQL scripts support fraud analysis and transaction analytics.

The dashboard uses prepared analytics files to display transaction statistics without requiring a live PostgreSQL connection in the deployed container.

## Machine Learning

The fraud detection model uses a Random Forest classifier.

**Model features:**
- Transaction type
- Transaction amount
- Sender's balance before the transaction
- Recipient's balance before the transaction

The model was trained on a reproducible sample of up to 300,000 transactions from the training partition, using `TRANSFER` and `CASH_OUT` transactions.

Data was separated chronologically using simulation time steps into approximately 60% training, 20% validation, and 20% testing.

Multiple classification thresholds were evaluated on the validation data.

### Model Evaluation Results

The saved Random Forest model was evaluated on 552,504 held-out transactions using a classification threshold of 0.80.

| Metric | Result |
|---|---:|
| Precision | 88.36% |
| Recall | 68.65% |
| Average Precision | 81.82% |
| Correctly detected fraud transactions | 2,923 |
| Missed fraud transactions | 1,335 |
| False positives | 385 |
| Correctly classified legitimate transactions | 547,861 |

These results apply to the synthetic PaySim dataset and should not be interpreted as validated performance on real-world banking transactions.

## Interactive Dashboard

The Streamlit application includes three pages:

1. **Overview:** Transaction and fraud statistics.
2. **Fraud Screening:** Model-based fraud-risk predictions for transaction inputs.
3. **Transaction Analytics:** Transaction patterns and fraud-related trends.

**Try the live dashboard:** https://ml-fraud-detection-d.duckdns.org

## Deployment Architecture

The application is deployed on AWS EC2 using Docker.

The deployed dashboard loads its saved Random Forest model and precomputed analytics files. Caddy acts as a reverse proxy and provides HTTPS access.

```text
PaySim Dataset
      |
      v
Python ETL Pipeline
      |
      v
PostgreSQL
      |
      +----> SQL Transaction Analytics
      |
      +----> Random Forest Training
                    |
                    v
              Saved ML Model
                    |
                    v
            Streamlit Dashboard
                    |
                    v
              Docker Container
                    |
                    v
                 AWS EC2
                    |
                    v
              Caddy / HTTPS
                    |
                    v
              Web Browser
```

The deployed dashboard does not require a live connection to PostgreSQL for predictions or prepared analytics.

## Project Structure

```text
ML_bankfrauddetection/
├── dashboard/
│   └── app.py
├── data/
│   └── dashboard/
├── models/
├── sql/
│   ├── fraud_analysis.sql
│   └── transaction_analysis.sql
├── src/
│   ├── etl/
│   └── ml/
├── .streamlit/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Reproducibility and Data

The repository excludes the large raw and processed datasets, saved machine learning models, environment secrets, and Docker image archives.

The ETL and model-training scripts are included. Reproducing the complete pipeline requires obtaining the PaySim dataset, configuring PostgreSQL, loading the data, and training the model.

## Limitations

- PaySim contains synthetic transactions rather than real customer banking records.
- Model performance may differ on real financial data.
- Fraud-risk predictions are experimental and should not be used as the sole basis for financial decisions.
- The deployed application is a portfolio demonstration, not a production banking fraud detection system.
