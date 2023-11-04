import pandas as pd


url = 'https://drive.google.com/uc?id=1S9Z38cffP2KSaXB-vW0BKpzSuWFhhKZM'
data = pd.read_csv(url)
data.to_csv('/home/clearml/projects/xFlow/mlops_practice_3/datasets/data.csv', sep=',', index = False)
