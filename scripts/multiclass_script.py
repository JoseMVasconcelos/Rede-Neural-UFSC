import kagglehub
import shutil
import os
import polars as pl
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

# Download dataset
path = kagglehub.dataset_download("prathamtripathi/drug-classification")
# Format downloaded files
for root, dirs, files in os.walk(path):
    for file in files:
        df = pl.read_csv(path+"/"+file)
        # Label encoding categorical attributes
        # Saving the old columns to act as reference the labels
        df = df.with_columns(
            pl.col("Sex").map_batches(enc.fit_transform).alias("Sex_encoded"),
            pl.col("Cholesterol").map_batches(enc.fit_transform).alias("Cholesterol_encoded"),
        )
        df = df.to_dummies("BP")
        # Saving file at the data folder
        df.write_csv("data/multiclass_drug.csv")
