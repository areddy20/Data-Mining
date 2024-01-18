# References used:
# 1) https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide#3.-Transactions
# 2) https://www.kaggle.com/code/robikscube/tutorial-time-series-forecasting-with-xgboost


import numpy as np
import pandas as pd
import os
import gc
import warnings

# PACF - ACF
# ------------------------------------------------------
import statsmodels.api as sm

# DATA VISUALIZATION
# ------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# MODEL
# ------------------------------------------------------
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error


# CONFIGURATIONS
# ------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')


# Import -------------------------------------------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
stores = pd.read_csv("stores.csv") 
transactions = pd.read_csv("transactions.csv").sort_values(["store_nbr", "date"])

# Datetime
train["date"] = pd.to_datetime(train.date)
test["date"] = pd.to_datetime(test.date)
transactions["date"] = pd.to_datetime(transactions.date)

# Data types
train.onpromotion = train.onpromotion.astype("float16")
train.sales = train.sales.astype("float32")
stores.cluster = stores.cluster.astype("int8")

# Did we get our data correctly?
print('Training data:\n' + str(train.head(10)))

print('Test Data:\n' + str(test.head(10)))



# Transactions -----------------------------------------------------
#temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how = "left")
#print("Spearman Correlation between Total Sales and Transactions: {:,.4f}".format(temp.corr("spearman").sales.loc["transactions"]))
#px.line(transactions.sort_values(["store_nbr", "date"]), x='date', y='transactions', color='store_nbr',title = "Transactions" )

#print('\nTransaction data:\n' + str(transactions.head(10)))


# Zero Forecasting (for product families that never sell because a store doesn't sell those products)
# c = train.groupby(["store_nbr", "family"]).sales.sum().reset_index().sort_values(["family","store_nbr"])
# c = c[c.sales == 0]

# # Anti Join
# outer_join = train.merge(c[c.sales == 0].drop("sales",axis = 1), how = 'outer', indicator = True)
# train = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis = 1)
# del outer_join
# gc.collect()

# zero_prediction = []
# for i in range(0,len(c)):
#     zero_prediction.append(
#         pd.DataFrame({
#             "date":pd.date_range("2017-08-16", "2017-08-31").tolist(),
#             "store_nbr":c.store_nbr.iloc[i],
#             "family":c.family.iloc[i],
#             "sales":0
#         })
#     )

# zero_prediction = pd.concat(zero_prediction)
# del c
# gc.collect()

#print('\nZero Forecasted Entries:\n' + str(zero_prediction))


# MODEL -------------------------------------------------------------------------------

def create_features(df, label=None):
    """
    Features to give the model for training.
    """
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day

    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth']]
    
    print(str(X))
    if label:
        y = df[label]
        return X, y
    return X


X_train, y_train = create_features(train, label='sales')
X_test = create_features(test)

print(str(X_train))


# train our model
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train)],
        early_stopping_rounds=50,
        verbose=False)

# predict for the test set
test['Prediction'] = reg.predict(X_test)
print(str(test['Prediction'].to_string()))
