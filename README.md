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

├── .github/workflows/        # CI/CD pipelines (feature, training, inference)

├── frontend/                 # Streamlit dashboards (monitoring & visualization)

│   ├── frontend_monitor.py

│   ├── frontend_v1.py

│   ├── frontend_v2.py

├── notebooks/                # Jupyter notebooks for data prep, EDA, and modeling

│   ├── 01_fetch_data.ipynb

│   ├── ... (baseline, XGBoost, LightGBM, hyperparameter tuning, retraining, etc.)

│   └── latest_prediction.csv

├── pipelines/                # ML pipelines (training & inference)

│   ├── model_training_pipeline.py

│   └── inference_pipeline.py

├── src/                      # Core source code & utilities

│   ├── config.py

│   ├── data_utils.py

│   ├── feature_pipeline.py

│   ├── inference.py

│   └── plot_utils.py

├── test/                     # Testing scripts

│   └── sample_app.py

├── requirements.txt          # Dependencies

├── requirements_feature_pipeline.txt

├── requirements_with_version.txt

├── todo.md                   # Pending tasks

└── vscode_config.json

**How to Run Locally**

**Clone the repo:**

git clone https://github.com/nivesharath/citi_bike.git

cd citi_bike

**Install dependencies:**

pip install -r requirements.txt

**Run the pipeline:**

python src/feature_pipeline.py

**Launch Streamlit dashboard:**

streamlit run frontend/frontend_v2.py

**Results**

1.Forecasted hourly demand trends with MAE < X rides/hour (update with your best metric).

2.Visualized predicted vs. actual rides for better demand planning.

- Trained models achieved consistent forecasting accuracy with MAE in the ~5–7 rides/hour range.  
- Demonstrated clear demand patterns across hours and weekdays.  
- Enabled comparison of predicted vs. actual rides for decision-making.  


**Future Improvements**

1.Incorporate weather & holiday data for richer forecasts.

2.Deploy a real-time API for serving predictions.
