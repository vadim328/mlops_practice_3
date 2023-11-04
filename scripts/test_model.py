import pandas as pd
import numpy as np
import os
import pickle
from catboost import CatBoostClassifier
from sklearn import metrics

cbc = pickle.load(open('/home/clearml/projects/xFlow/mlops_practice_3/models/model.pkl', 'rb'))
test_df = pd.read_csv('/home/clearml/projects/xFlow/mlops_practice_3/datasets/data_test.csv', sep=',')

X = test_df.drop("outcome", axis = 1)
y = test_df['outcome']

y_pred = cbc.predict(X)

score = metrics.accuracy_score(y, y_pred)

print("score=", score)
