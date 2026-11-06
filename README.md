# Banking Transaction Intelligence Platform

An end-to-end **Data Engineering + Data Science** portfolio project for transaction analytics and anomaly detection.

## Status
**In development.** This repository is a project scaffold, not yet a working or deployed application. No model accuracy or fraud-detection performance is claimed.

## Intended architecture
1. Ingest a documented public or synthetic transaction dataset.
2. Validate and transform transactions through a reproducible Python ETL pipeline.
3. Load raw and curated records into PostgreSQL.
4. Analyze transaction trends using SQL and Python.
5. Train and evaluate an anomaly-detection model; distinguish anomalies from confirmed fraud.
6. Present insights in a Streamlit dashboard.
7. Add tests, Docker, CI, and optionally deploy to AWS after local validation.

## Planned stack
Python, pandas, PostgreSQL, SQL, scikit-learn, Streamlit, pytest, Docker, GitHub Actions, AWS.

## Data and security
- No real customer financial records or credentials should be committed.
- Dataset provenance, license, fields, and limitations will be documented before ingestion.
- Any synthetic records will be clearly labeled synthetic.
- Model evaluation will report limitations and avoid unsupported fraud-detection claims.

## Development milestones
- [ ] Select and document dataset
- [ ] Implement reproducible ETL and validation
- [ ] Define PostgreSQL schema and load pipeline
- [ ] Build exploratory analysis and SQL metrics
- [ ] Train and evaluate anomaly detection
- [ ] Create Streamlit dashboard
- [ ] Add automated tests and CI
- [ ] Containerize and deploy
- [ ] Publish screenshots and results
