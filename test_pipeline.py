import pandas as pd
import numpy as np

def clean_data(df):
    df = df.copy()
    
    none_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
                 "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                 "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", 
                 "BsmtFinType2", "MasVnrType"]
    for col in none_cols:
        df[col] = df[col].fillna("None")
    
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
    
    return df

def test_clean_data_no_missing_values():
    df = pd.read_csv("data/train.csv")
    df_clean = clean_data(df)
    assert df_clean.isnull().sum().sum() == 0, "There are still missing values!"

def test_clean_data_returns_dataframe():
    df = pd.read_csv("data/train.csv")
    df_clean = clean_data(df)
    assert isinstance(df_clean, pd.DataFrame)