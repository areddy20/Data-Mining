from hashlib import sha1
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import tensorflow as tf
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
#store_nbr is store at which products are sold
#family is type of product
#sales = total sales for product type at given date
#onpromotion = number of items in product family on promotion

x = 10
train = pd.read_csv('train.csv')
train = train.iloc[:len(train) // x]
holidays_events = pd.read_csv('holidays_events.csv')
holidays_events = holidays_events.iloc[:len(holidays_events) // x]
oil_data = pd.read_csv('train.csv')
oil_data = oil_data.iloc[:len(oil_data) // x]
stores = pd.read_csv('train.csv')
stores = stores.iloc[:len(stores) // x]
transactions = pd.read_csv('transactions.csv')
transactions = transactions.iloc[:len(transactions) // x]
test = pd.read_csv('test.csv')
test = test.iloc[:len(test) // x]
print(holidays_events.shape)
print(oil_data.shape)
print(stores.shape)
print(transactions.shape)
print(train.shape)
print(test.shape)



train["test"] = 0
test["test"] = 1
df = pd.concat([train, test], axis=0)

print(df)

def add_datetime_features(df):
    
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.day_of_week
    df["day_name"] = df["date"].dt.day_name()
    df["quarter"] = df["date"].dt.quarter
    df["is_leap_year"] = df["date"].dt.is_leap_year
    return df
df = add_datetime_features(df)
print(df.shape)
print("df_shape")

holidays_events["date"] = pd.to_datetime(holidays_events["date"])
transactions["date"] = pd.to_datetime(transactions["date"])
df = pd.merge(df, stores, on="store_nbr", how="left")
df = pd.merge(df, holidays_events, on="date", how="left")

df.drop("description", inplace=True, axis=1)
df.drop("date", inplace=True, axis=1)
df.drop("type_x", inplace=True, axis=1)
df.drop("state", inplace=True, axis=1)
df.drop("day_name", inplace=True, axis=1)

categorical_columns = ["family","city","type_y","locale","locale_name", "transferred"]
label_encode_cols = categorical_columns
label_encoder = LabelEncoder()
print("df_copy")
final_df = df.copy()

final_df = train = train.iloc[:len(train) // 50]



for col in label_encode_cols:
    final_df[col] = label_encoder.fit_transform(final_df[col])
    
print(final_df.head())