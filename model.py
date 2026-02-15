"""
XGBoost-based model for training and predicting prices in Path Of Exile.

This module contains functions to train an XGBoost regression model
and generate price predictions from input features.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

CURRENCIES = [
    "Orb of Transmutation",
    "Orb of Augmentation",
    "Orb of Alteration",
    "Chromatic Orb",
    "Jeweller's Orb",
    "Orb of Fusing",
    "Orb of Chance",
    "Orb of Alchemy",
    "Orb of Scouring",
    "Blessed Orb",
    "Regal Orb",
    "Gemcutter's Prism",
    "Glassblower's Bauble",
    "Exalted Orb",
    "Divine Orb",
    "Vaal Orb",
    "Orb of Regret",
    "Orb of Annulment",
    "Orb of Binding",
    "Ancient Orb",
    "Harbinger's Orb",
    "Fracturing Orb",
    "Sacred Orb",
    "Enkindling Orb",
    "Instilling Orb",
    "Orb of Horizons",
    "Orb of Unmaking",
    "Mirror of Kalandra",
    "Hinekora's Lock",
    "Mirror Shard",
    "Awakener's Orb",
    'Veiled Exalted Orb',
    'Veiled Chaos Orb',
    'Orb of Dominance',
    "Hunter's Exalted Orb",
    "Warlord's Exalted Orb",
    "Crusader's Exalted Orb",
    "Redeemer's Exalted Orb",
    'Orb of Conflict',
    'Exceptional Eldritch Ichor',
    'Exceptional Eldritch Ember'
]

FEATURES = [
    'days_in_league',
    'dow',
    'lag_1',
    'lag_3',
    'lag_7',
    'rolling_mean_3',
    'rolling_mean_7',
    'rolling_std_7'
    ]


def train(df:pd.DataFrame, model:XGBRegressor) -> XGBRegressor:
    # make training set
    X = df[FEATURES]
    y = df['log_price']
 
    # train model
    model.fit(X,y)

    # check feature dominance
    print(f'Feature importance: {pd.Series(model.feature_importances_,index=FEATURES).sort_values(ascending=False)}')

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
        'dow': last['dow'] + 1 if last['dow'] != 6 else 0,
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
    
def validate_model(df:pd.DataFrame,model:XGBRegressor) -> dict[float,float]:
    #init vars 
    decay_grid = [0.0, 0.3, 0.6, 1.0]
    leagues = df['League'].unique()
    results = {}
    # loop over decay grid for league-age weighting
    for decay in decay_grid:
        # calculate weights
        df['sample_weight'] = np.exp(-decay * df['league_age'])
        mses = []
        
        # holdout loop with one league left out
        for holdout in leagues:
            # test/train split
            train = df[df['League'] != holdout].copy()
            test = df[df['League'] == holdout].copy()

            # train and test
            model.fit(
                train[FEATURES],
                train['log_price'],
                sample_weight=train['sample_weight']
            )
            
            # record MAE for league
            pred = np.expm1(model.predict(test[FEATURES]))
            mses.append(mean_squared_error(test['Price'],pred))
        # output and save mean MSE for this decay
        results[decay] = np.mean(mses)
        print(f'Decay: {decay:.2f} => MSE: {results[decay]:.4f}')
    # output best MSE:decay (min value)
    print(f'Best Decay: {min(results, key=results.get)}')
    return results

def main():
    # read data and prep for use
    df = pd.read_csv('datasets/master_set.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    # only get the currencies we want
    #df = df[df['item_name'].isin(CURRENCIES)]
    df = df[df['item_name'] == 'Divine Orb']

    # init model
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
    #validate model
    validate_model(df, model)
    # TODO validate_model needs to do check other param values and find best

if __name__ == '__main__':
    main()
