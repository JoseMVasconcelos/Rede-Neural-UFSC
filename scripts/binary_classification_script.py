import kagglehub
import os
import polars as pl

file_path = ''
path = kagglehub.dataset_download('rabieelkharoua/predict-customer-purchase-behavior-dataset')

for root, dirs, files in os.walk(path):
    for file in files:
        df = pl.read_csv(path + '/' + file)
        df.write_csv('data/customer-purchase-behavior.csv')
