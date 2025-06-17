import kagglehub
import os
import polars as pl
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

file_path = ""

path = kagglehub.dataset_download("vipullrathod/fish-market");

for root, dirs, files in os.walk(path):
  for file in files:
      df = pl.read_csv(path+"/"+file)
      df = df.to_dummies("Species")
      df.write_csv("data/regression_fish.csv")