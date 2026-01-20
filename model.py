"""
XGBoost-based model for training and predicting prices in Path Of Exile.

This module contains functions to train an XGBoost regression model
and generate price predictions from input features.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor



def train(df:pd.DataFrame) -> XGBRegressor:
    
    # set features and target
    features = [
    'days_in_league',
    'lag_1',
    'lag_3',
    'lag_7',
    'rolling_mean_3',
    'rolling_mean_7',
    'rolling_std_7'
    ]

    # make training set
    X = df[features]
    y = df['log_price']

    # init model
    # TODO:try with reg:pseudohubererro as alt maybe?
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        gamma=0.1,
        random_state=42,
        tree_method='hist',
    )
    # train model
    model.fit(X,y)

    return model

def predict_tomorrow(df:pd.DataFrame, model:XGBRegressor, item_name:str, league:str) -> float:
    # get data for item and current league (recommended)
    item_df = df[
        (df['item_name'] == item_name) &
        (df['League'] == league)
    ].sort_values('Date')

    if len(item_df) == 0:
        raise ValueError(f'No data for {item_name} in {league}')
    
    last = item_df.iloc[-1]

    # Make tomorrow frame
    X_next = pd.DataFrame([{
        'days_in_league': last['days_in_league'] + 1,
        'lag_1': last['Price'],
        'lag_3': item_df['Price'].iloc[-3] if len(item_df) >= 3 else np.nan,
        'lag_7': item_df['Price'].iloc[-7] if len(item_df) >= 7 else np.nan,
        'rolling_mean_3': item_df['Price'].iloc[-3:].mean() if len(item_df) >= 3 else np.nan,
        'rolling_mean_7': item_df['Price'].iloc[-7:].mean() if len(item_df) >= 7 else np.nan,
        'rolling_std_7': item_df['Price'].iloc[-7:].std() if len(item_df) >= 7 else np.nan,
    }])

    # predict 'log price' then return after conversion back
    log_pred = model.predict(X_next)[0]
    return np.expm1(log_pred)
    

def main():
    # read data and prep for use
    df = pd.read_csv('datasets/master_set.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    #TODO: add logic to pick training or predicting
    # add saving model locally 
    # add output price predications to plot
    # add code to get latest prices for more upto date item_df

