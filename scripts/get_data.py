import pandas as pd
import os

import mlflow
from mlflow.tracking import MlflowClient


os.environ["MLFLOW_REGISTRY_URI"] = "/home/clearml/projects/xFlow/mlops_practice_3/scripts/mlflow"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")

url = 'https://drive.google.com/uc?id=1S9Z38cffP2KSaXB-vW0BKpzSuWFhhKZM'
with mlflow.start_run():
    data = pd.read_csv(url)
    mlflow.log_artifact(local_path="/home/clearml/projects/xFlow/mlops_practice_3/scripts/get_data.py",
                        artifact_path="get_data code")
    mlflow.end_run()
data.to_csv('/home/clearml/projects/xFlow/mlops_practice_3/datasets/data.csv', sep=',', index = False)
