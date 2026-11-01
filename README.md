# Self-Healing Machines: Predictive Maintenance (Teliport S4, Case Study 1)

Team Innovatrix, KIET Group of Institutions (3rd year):
Anshu Kumari (Leader), Anshul Anand, Simmi Kumari

Closed loop from sensor data to maintenance work order on all four NASA C-MAPSS subsets:
data engineering, RUL prediction with 90% range, anomaly detection with root cause,
rule-based triage agent, gated MLOps retraining, Streamlit dashboard.

## Run

    pip install -r requirements.txt
    python run_pipeline.py
    streamlit run dashboard/app.py

## Structure

- src/ : data, features, models, anomaly, training, agent, MLOps
- dashboard/app.py : monitoring dashboard
- notebooks/ : step-by-step analysis
- reports/ : result tables and figures
- work_orders/ : generated work orders

Dataset: NASA C-MAPSS (https://data.nasa.gov)

## Sensor fault detection (upgrade)

The pipeline can flag a broken sensor, name the suspect sensor and fall back to a backup RUL
estimate computed without it. See `11_upgrades.ipynb` for the demo (10 engines, one broken sensor each).
