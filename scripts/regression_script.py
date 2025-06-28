import kagglehub
import os
import polars as pl
from sklearn.preprocessing import StandardScaler

file_path = ""

path = kagglehub.dataset_download("vipullrathod/fish-market")

for root, dirs, files in os.walk(path):
  for file in files:
      df = pl.read_csv(path+"/"+file)

      # One-hot-encoding for fish species.
      df_onehot = df.to_dummies("Species")
      
      # Columns with numerical values.
      numerical_cols = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
      numerical_data = df_onehot[numerical_cols].to_numpy()
      # Scaling the data
      scaler = StandardScaler()
      scaled_data = scaler.fit_transform(numerical_data)
      
      df_scaled = pl.DataFrame(scaled_data, schema=numerical_cols)
      
      # Updating data.
      df_final = df_onehot.update(df_scaled)
      df_final.write_csv("data/regression_fish.csv")