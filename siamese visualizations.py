import pandas as pd
import numpy as np
import regex as re
import gzip
import csv
import json
import torch
import tensorflow as tf
import pickle
import requests
import re

#defining data_types per column and load data function
COLUMN_REGEXES = {
    re.compile(r'user_game_win_rate_bucket'): 'float16',
    re.compile(r'user_n_games_bucket'): 'int8',
    re.compile(r'draft_id'): 'str',
    re.compile(r'draft_time'): 'str',
    re.compile(r'expansion'): 'str',
    re.compile(r'event_type'): 'str',
    re.compile(r'event_match_wins'): 'int8',
    re.compile(r'event_match_losses'): 'int8',
    re.compile(r'pack_number'): 'int8',
    re.compile(r'pick_number'): 'int8',
    re.compile(r'pick'): 'str',
    re.compile(r'pick_maindeck_rate'): 'float16',
    re.compile(r'pick_sideboard_in_rate'): 'float16',

    re.compile(r'pool_.*'): 'int8',
    re.compile(r'pack_card_.*'): 'int8',
}

def load_data(filename):
    col_names = pd.read_csv(filename, nrows=0).columns
    data_types = {}
    for c in col_names:
        for (r, t) in COLUMN_REGEXES.items():
            if r.match(c):
                data_types[c] = t
    skipcols= ['draft_time',
               'event_type',
               'expansion',
               'event_match_wins',
               'event_match_losses',
               'user_n_games_bucket',
               'user_game_win_rate_bucket',
               'pick_maindeck_rate',
               'pick_sideboard_in_rate',
               'draft_id',
               #'pick_number',
               #'pack_number',
               'rank'
                ]
    df = pd.read_csv(
        filename,
        dtype=data_types,
        #nrows=100000,
        #skiprows=range(1, 5000000),
        #chunksize=10000,
        usecols = lambda x: x not in skipcols
        #usecols = ['rank', 'pack_number', 'pick_number']
    )
      
    
    return df

# API request form taken from Ryan Saxe
def get_card_rating_data(expansion, endpoint=None, start="2022-09-01", end="2023-05-24", colors=None):
    if endpoint is None:
        endpoint = f"https://www.17lands.com/card_ratings/data?expansion={expansion.upper()}&format=PremierDraft"
        if start is not None:
            endpoint += f"&start_date={start}"
        if end is not None:
            endpoint += f"&end_date={end}"
        if colors is not None:
            endpoint += f"&colors={colors}"
    card_json = requests.get(endpoint).json()
    card_df = pd.DataFrame(card_json).fillna(0.0)
    #numerical_cols = card_df.columns[card_df.dtypes != object]
    #card_df["name"] = card_df["name"].str.lower()
    card_df = card_df.set_index("name")
    return card_df[["rarity", "color"]]

def replace_colors(match):
    color_mapping = {'G': 'Green', 'R': 'Red', 'B': 'Black', 'U': 'Blue', 'W': 'White'}
    colors = match.group(0)
    replacements = [color_mapping.get(color, color) for color in colors]
    return ', '.join(replacements)

def add_stats(rate_df, stats_df):
# adding basic lands
  basic_lands = pd.DataFrame({
    'rarity': ['C', 'C', 'C', 'C', 'C'],
    'color': ['', '', '', '', '']}, 
    index=['Plains', 'Island', 'Swamp', 'Mountain', 'Forest'])

  stats_df = pd.concat([stats_df, basic_lands])

# renaming rarities
  rarities = {'uncommon': 'U', 'rare': 'R', 'common': 'C', 'mythic': 'M', 'basic':'C'}
  stats_df["rarity"] = stats_df['rarity'].replace(rarities)

# apply color replacement
  stats_df['color'] = stats_df['color'].str.replace(r'[GWRBU]+', replace_colors, regex=True)
  stats_df['color'] = stats_df['color'].replace('', 'Colourless')

# final concat
  final_df = pd.concat([rate_df, stats_df], axis=1)
  return final_df

# Creating the pick rate file

def get_pickrate(df, path):
# Step 1: Count the number of times each card was picked
  pick_counts = df['pick'].value_counts().sort_index()

# Step 2: Count the number of times each card was possible to be picked
  pack_cols = df.columns[df.columns.str.startswith('pack_card_')]
  possible_counts = df[pack_cols].sum()
  possible_counts.index = possible_counts.index.str.replace('pack_card_', '')

# Step 3: Calculate the pick rate
  pick_rate = pick_counts / possible_counts

# Step 4: Create pickrate file
  rate_df = pd.concat([pick_rate.rename('pick_rate'), pick_counts.rename('pick_count')], axis=1)
  rate_df['pick_rate'] = rate_df['pick_rate'].fillna(0)
  rate_df['pick_count'] = rate_df['pick_count'].fillna(0)

# Step 5: Get rarity and color for graphics/tables
  stats_df = get_card_rating_data("DMU")
  final_df = add_stats(rate_df, stats_df)

# Step 6: Save to csv
  final_df.to_csv(path, index=True, header=False)

def compute_pick_chance(dataset):
  df = load_data(dataset)
  other_cols = np.array(df.columns[0:3])
  sorted_cols = np.sort(df.columns[3:].tolist())
  sorted_cols = np.insert(sorted_cols, 0, other_cols, axis=0)
  df = df.loc[:, sorted_cols]
  get_pickrate(df, 'pickrate_full.csv')

def compute_firstpick_chance(dataset):
  df = load_data(dataset)
  other_cols = np.array(df.columns[0:3])
  sorted_cols = np.sort(df.columns[3:].tolist())
  sorted_cols = np.insert(sorted_cols, 0, other_cols, axis=0)
  df = df.loc[:, sorted_cols]
  df["pick_number"] = df["pick_number"].astype("int8")

# Step 1: Filter df to only get first picks (pack_number AND pick_number are both 0)
  filtered_df = df[(df['pack_number'] == 0) & (df['pick_number'] == 0)]
# Step 2: Get first pick rates
  get_pickrate(filtered_df, 'firstpickrate_full.csv')

compute_pick_chance(r"Data\DMU_draft_data.csv")
compute_firstpick_chance(r"Data\DMU_draft_data.csv")