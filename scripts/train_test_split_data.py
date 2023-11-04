import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


  
data = pd.read_csv('/home/clearml/projects/xFlow/mlops_practice_3/datasets/data_processed.csv', sep=',')

X = data.drop("outcome", axis = 1)
y = data['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = X_train.join(y_train)
test_df = X_test.join(y_test)

train_df.to_csv('/home/clearml/projects/xFlow/mlops_practice_3/datasets/data_train.csv', sep=',', index = False)
test_df.to_csv('/home/clearml/projects/xFlow/mlops_practice_3/datasets/data_test.csv', sep=',', index = False)
