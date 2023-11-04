import pandas as pd
import numpy as np
import os
import pickle
from catboost import CatBoostClassifier


train_df = pd.read_csv('/home/clearml/projects/xFlow/mlops_practice_3/datasets/data_train.csv', sep=',')

X = train_df.drop("outcome", axis = 1)
y = train_df['outcome']

n_estimators = 300

cbc = CatBoostClassifier(verbose=0, n_estimators=n_estimators)
cbc.fit(X, y)

with open("/home/clearml/projects/xFlow/mlops_practice_3/models/model.pkl", "wb") as f:
    pickle.dump(cbc, f)
