"""
This script processes raw economy data from poe.ninja and prepares it for modeling.

It ingests both historic datasets and current active league data:
- Historic data is read from datasets/raw/
- Current data is fetched from the poe.ninja api

The script cleans and reshapes the data into a long format suitable for
league-by-league modeling with XGBoost. Also generates lag and rolling mean features
...
Network access is required to fetch current league data.
"""

import os
import pandas as pd
import numpy as np
import requests
import time


def get_current_prices(league:str) -> dict[str, float]:
    """Pulls today's prices for each currency item from poe.ninja"""
    # pull data from api
    url = f'https://poe.ninja/poe1/api/economy/exchange/current/overview?league={league}&type=Currency'
    res = requests.get(url)

    if res.status_code == 200:
        data = res.json()
        prices = {}
        # for each currency we want the real name and its value
        for line, item in zip(data['lines'],data['items']):
            prices[item['name']] = line['primaryValue']
        # this is always 1 so drop it (what the price is measured in)
        del prices['Chaos Orb']
        return prices
    else:
        print(f'Request failed with code: {res.status_code}')
        return {}


def get_league_history(league:str) -> pd.DataFrame:
    """Gets the history from league start to current day for all currency items of a valid active league on poe.ninja"""
    # pull data from api to get available currency prices
    url = f'https://poe.ninja/poe1/api/economy/exchange/current/overview?league={league}&type=Currency'
    res = requests.get(url)
    data = {'League':league,
            'Date': [],
            'item_name':[],
            'Price':[]}
    if res.status_code == 200:
        # get the id's for second api call to individual history
        ids = {item['detailsId']:item['name'] for item in res.json()['items']}
        # remove chaos orb
        ids.pop('chaos-orb')
        for id in ids.keys():
            url = f'https://poe.ninja/poe1/api/economy/exchange/current/details?league={league}&type=Currency&id={id}'
            res = requests.get(url)

            if res.status_code == 200:
                # grab the history
                history = res.json()['pairs'][0]['history']
                name = ids[id]
                # pull day, name and price into separate ordered tuples 
                days, item_name, prices = zip(*((day['timestamp'][:10],name,day['rate']) for day in history))

                # append the data to the lists
                data['Date'] += list(days)
                data['item_name'] += list(item_name)
                data['Price'] += list(prices)
            else:
                print(f'Request for {ids[id]} failed with code: {res.status_code}')

            # rate limit a bit
            time.sleep(0.1)

        # make frame
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])

        # sort it
        df.sort_values(['item_name','Date'], inplace=True, ignore_index=True)
        
        # add number of days the league has been going on
        league_start = df['Date'].min()
        df['days_in_league'] = (df['Date'] - league_start).dt.days
        
        # create lag features and rolling mean features
        for lag in [1,3,7]:
            df[f'lag_{lag}'] = df.groupby('item_name')['Price'].shift(lag)

        df['rolling_mean_3'] = df.groupby('item_name')['Price'].shift(1).rolling(3).mean()

        df['rolling_mean_7'] = df.groupby('item_name')['Price'].shift(1).rolling(7).mean()
        
        df['rolling_std_7'] = df.groupby('item_name')['Price'].shift(1).rolling(7).std()
    
        # log transform target
        df["log_price"] = np.log1p(df["Price"])
        
        # return frame
        return df
    else:
        print(f'Request failed with code: {res.status_code}')
        return pd.DataFrame(data) 


def update_master_set():
    """
    Takes the raw economy data dump from poe.ninja (local files), cleans it up and adds features for model training.
    Creates/Updates 'master_set.csv' which contains data from all leagues including current active league.
    """
    # main dataframe
    df_out = pd.DataFrame()
    for filename in os.listdir('datasets/raw/'):
        # load csv
        df = pd.read_csv(f'datasets/raw/{filename}',sep=';')

        # set date right
        df['Date'] = pd.to_datetime(df['Date'])

        # remove the bad data (this is junk)
        df = df[~df['Get'].str.contains('chaos orb', case=False, na=False)]

        # filter by confidence
        df = df[df['Confidence'].str.contains('High', case=False, na=False)]

        # drop useless data
        df = df.drop(columns=['Confidence','Pay'])

        # rename and sort
        df.rename(columns={'Get':'item_name', 'Value':'Price'}, inplace=True)
        df.sort_values(['item_name','Date'], inplace=True, ignore_index=True)
        
        # add number of days the league has been going on
        league_start = df['Date'].min()
        df['days_in_league'] = (df['Date'] - league_start).dt.days

        # create lag features and rolling mean features
        for lag in [1,3,7]:
            df[f'lag_{lag}'] = df.groupby('item_name')['Price'].shift(lag)

        df['rolling_mean_3'] = df.groupby('item_name')['Price'].shift(1).rolling(3).mean()

        df['rolling_mean_7'] = df.groupby('item_name')['Price'].shift(1).rolling(7).mean()
        
        df['rolling_std_7'] = df.groupby('item_name')['Price'].shift(1).rolling(7).std()
    
        # log transform target
        df["log_price"] = np.log1p(df["Price"])

        # add to out
        df_out = pd.concat([df_out,df], ignore_index=True)

    # get active league data and add it to out
    df_current = get_league_history('Keepers')
    df_out = pd.concat([df_out,df_current], ignore_index=True)

    # add day of week
    df_out['dow'] = df_out['Date'].dt.weekday

    # add league age for weighting
    league_order = (df_out[['League', 'Date']].groupby('League')['Date'].min().sort_values().index.tolist())
    league_age = {league:age for age, league in enumerate(league_order[::-1])}
    df_out['league_age'] = df_out['League'].map(league_age)

    # save in csv
    df_out.to_csv('datasets/master_set.csv', index=False)

if __name__ == '__main__':
    update_master_set()
