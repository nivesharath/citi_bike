import hopsworks
import joblib
import os
import pandas as pd
from pathlib import Path
from datetime import timedelta

from hsfs.feature_store import FeatureStore

import src.config as config
from src.data_utils import transform_ts_data_info_features


def get_hopsworks_project() -> hopsworks.project.Project:
    """Log in to Hopsworks and return the project handle."""
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )


def get_feature_store() -> FeatureStore:
    """Grab the Feature Store client from your Hopsworks project."""
    project = get_hopsworks_project()
    return project.get_feature_store()


def load_model_from_registry(model_name: str = None, version: int = None):
    """
    Download & load the latest sklearn Pipeline you registered in Hopsworks.
    Returns a joblib-loaded pipeline object.
    """
    project        = get_hopsworks_project()
    registry       = project.get_model_registry()
    models         = registry.get_models(name = model_name or config.MODEL_NAME)
    best           = max(models, key=lambda m: m.version if version is None else (m.version == version))
    download_dir   = best.download()
    
    # Changed model filename to match your local file
    artifact_path  = Path(download_dir) / "xgb_model.pkl"
    
    return joblib.load(artifact_path)



def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """Apply model to features and return prediction DataFrame."""
    preds_array = model.predict(features)
    out = pd.DataFrame({
        "pickup_location_id": features["pickup_location_id"].values,
        "predicted_demand": preds_array.round(0).astype("int32")
    })
    return out


def main():
    fs = get_feature_store()
    pipeline = load_model_from_registry()

    # Set historical anchor time manually
    base_hour = pd.Timestamp("2023-12-31 18:00:00", tz="UTC")
    window_size = 24  # 1-day lookback for simplicity

    # Feature view and prediction FG
    fv = fs.get_feature_view(name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION)
    pred_fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTION,
        version=config.FEATURE_GROUP_MODEL_PREDICTION_VERSION,
        description="Next-hour predictions",
        primary_key=["pickup_location_id", "pickup_hour"],
        event_time="pickup_hour",
        online_enabled=False,
        features=[
            Feature("pickup_location_id", "string"),
            Feature("pickup_hour", "timestamp"),
            Feature("predicted_rides", "int"),
        ]
    )

    for i in range(5):  # Predict for 5 hours
        current_hour = base_hour + timedelta(hours=i)

        ts = fv.get_batch_data(
            start_time=current_hour - timedelta(hours=window_size),
            end_time=current_hour
        )

        if ts.empty:
            print(f"⚠️ No data found for {current_hour}, skipping.")
            continue

        ts = ts.sort_values(["pickup_location_id", "pickup_hour"])

        features = transform_ts_data_info_features(
            ts,
            feature_col="rides",
            window_size=window_size,
            step_size=1
        )
        features["target"] = 0

        preds = get_model_predictions(pipeline, features)
        preds = preds.rename(columns={"predicted_demand": "predicted_rides"})
        preds["pickup_hour"] = current_hour + timedelta(hours=1)
        preds["pickup_location_id"] = preds["pickup_location_id"].astype(str)
        preds["predicted_rides"] = preds["predicted_rides"].astype("int32")

        pred_fg.insert(preds, write_options={"wait_for_job": False})
        print("✅ Inserted prediction for:", preds["pickup_hour"].iloc[0])


if __name__ == "__main__":
    main()
