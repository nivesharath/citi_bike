# Citi Bike Trip Prediction System

**Project Overview**

This project forecasts hourly Citi Bike ride demand in New York City using historical trip data. The goal is to help bike-sharing operators and city planners optimize bike availability and rebalancing strategies.


**Features**

1.Built an end-to-end ML pipeline for time-series forecasting.

2.Created lag features & rolling statistics to capture demand trends.

3.Trained multiple models (Baseline, Lag-28, Feature-Selected) using LightGBM.

4.Logged experiments and metrics using MLflow.

5.Automated workflows via GitHub Actions for CI/CD.

6.Deployed predictions and features to Hopsworks Feature Store.

7.Built an interactive Streamlit dashboard for visualization & monitoring.


**Tech Stack**

**Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, LightGBM, Matplotlib

**ML Ops Tools:** MLflow, Hopsworks, GitHub Actions

**Visualization:** Streamlit

**Version Control:** GitHub


**Repository Structure**

├── notebooks/           # Exploratory Data Analysis & experiments

├── pipelines/           # ML pipeline scripts

├── src/                 # Core source code

├── frontend/            # Streamlit app for visualization

├── test/                # Unit tests

├── requirements.txt     # Dependencies

└── .github/workflows/   # CI/CD pipelines


**How to Run Locally**

**Clone the repo:**

git clone https://github.com/nivesharath/citi_bike.git

cd citi_bike

**Install dependencies:**

pip install -r requirements.txt

**Run the pipeline:**

python src/run_pipeline.py

**Launch Streamlit dashboard:**

streamlit run frontend/app.py

**Results**

1.Forecasted hourly demand trends with MAE < X rides/hour (update with your best metric).

2.Visualized predicted vs. actual rides for better demand planning.

- Trained models achieved consistent forecasting accuracy with MAE in the ~5–7 rides/hour range.  
- Demonstrated clear demand patterns across hours and weekdays.  
- Enabled comparison of predicted vs. actual rides for decision-making.  


**Future Improvements**

1.Incorporate weather & holiday data for richer forecasts.

2.Deploy a real-time API for serving predictions.
