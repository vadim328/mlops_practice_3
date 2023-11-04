import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
import os


def work_fith_data(df, key):

  # Применим степенное преобразование для числовых данных
  num_columns = list((df.select_dtypes(include=[int, float]).columns))
  pwt = PowerTransformer()
  num_df = pd.DataFrame(pwt.fit_transform(df[num_columns]))
  num_df.columns = num_columns

  category_columns = list((df.select_dtypes(include=[object]).columns))
  if key == 0: # One hot encoding для категориальных данных
    ohe = OneHotEncoder(handle_unknown='ignore')
    cat_df = pd.DataFrame(ohe.fit_transform(df[category_columns]).toarray())
  elif key == 1: # Ordinal encoding для категориальных данных
    ore = OrdinalEncoder()
    cat_df = pd.DataFrame(ore.fit_transform(df[category_columns]))
    cat_df.columns = category_columns

  return num_df.join(cat_df)
  
train_df = pd.read_csv('/home/clearml/projects/xFlow/mlops_practice_3/datasets/data.csv', sep=',')

train_df = work_fith_data(train_df, 1) # датафрейм для обучения

train_df.to_csv('/home/clearml/projects/xFlow/mlops_practice_3/datasets/data_processed.csv', sep=',', index = False)
