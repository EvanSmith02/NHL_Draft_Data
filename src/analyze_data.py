import requests
import re
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
import sklearn.metrics as skm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as pre
import os

def get_forwards():
    return pd.read_csv('data/processed/forward_merged_stats.csv')

def get_defenders():
    return pd.read_csv('data/processed/defender_merged_stats.csv')

def get_goalies():
    return pd.read_csv('data/processed/goalie_merged_stats.csv')

def predict_forward_round():
    forward_df = get_forwards()
    print(forward_df.columns)
    key_stats = forward_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'Seasons']]
    y = forward_df["Draft Position"]
    model = lm.LinearRegression()
    model.fit(key_stats, y)
    y_pred = model.predict(key_stats)
    print(y_pred)
    mse = skm.mean_squared_error(y, y_pred)
    r2 = skm.r2_score(y, y_pred)
    mae = skm.mean_absolute_error(y, y_pred)
    return y_pred

def forward_key_averages_per_round():
    forward_df = get_forwards()
    key_stats = forward_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
            averages[round][stat] = (forward_df[forward_df["Round"] == round][stat] / forward_df[forward_df["Round"] == round]["Seasons"]).fillna(0).mean()
    averages_df = pd.DataFrame(averages).T
    print("Forward Averages Per Round")
    print(averages_df)
    return averages
    
def defender_key_averages_per_round():
    defender_df = get_defenders()
    key_stats = defender_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        #averages[round]['GP_y'] = (defender_df[defender_df["Round"] == round]['GP_y'] / defender_df[defender_df["Round"] == round]["Seasons"]).mean()
        for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
            averages[round][stat] = (defender_df[defender_df["Round"] == round][stat] / defender_df[defender_df["Round"] == round]["Seasons"]).fillna(0).mean()
    averages_df = pd.DataFrame(averages).T
    print("Defender Averages Per Round")
    print(averages_df)
    return averages

def goalie_key_averages_per_round():
    goalie_df = get_goalies()
    goalie_df = goalie_df[goalie_df["GP_x"] != 0]
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        for stat in ['GS', 'W', 'L', 'T/O']:
            averages[round][stat] = (goalie_df[goalie_df["Round"] == round][stat] / goalie_df[goalie_df["Round"] == round]["Seasons"]).fillna(0).mean()
        for stat in ['GA', 'Shots', 'SV', 'SV%' ,'GAA']:
            averages[round][stat] = (goalie_df[goalie_df["Round"] == round][stat] / goalie_df[goalie_df["Round"] == round]["GP_x"]).mean()
    averages_df = pd.DataFrame(averages).T
    print("Goalie Averages per round")
    print(averages_df)
    return averages

def proportion_of_single_appearances():
    forward_df = get_forwards()
    defender_df = get_defenders()
    goalie_df = get_goalies()
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        averages[round]['forwards'] = len(forward_df[(forward_df["GP_y"] != 0) & (forward_df["Round"] == round)]) / len(forward_df[forward_df["Round"] == round])
        averages[round]['defenders'] = len(defender_df[(defender_df["GP_y"] != 0) & (defender_df["Round"] == round)]) / len(defender_df[defender_df["Round"] == round])
        averages[round]['goalie'] = len(goalie_df[(goalie_df["GP_y"] != 0) & (goalie_df["Round"] == round)]) / len(goalie_df[goalie_df["Round"] == round])
    averages_df = pd.DataFrame(averages).T
    print("Probability of player getting at least one appearance")
    print(averages_df)

def proportion_of_lasting_appearances():
    forward_df = get_forwards()
    defender_df = get_defenders()
    goalie_df = get_goalies()
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        averages[round]['forwards'] = len(forward_df[(forward_df["GP_y"] > 42) & (forward_df["Round"] == round)]) / len(forward_df[forward_df["Round"] == round])
        averages[round]['defenders'] = len(defender_df[(defender_df["GP_y"] > 42) & (defender_df["Round"] == round)]) / len(defender_df[defender_df["Round"] == round])
        averages[round]['goalie'] = len(goalie_df[(goalie_df["GP_y"] > 21) & (goalie_df["Round"] == round)]) / len(goalie_df[goalie_df["Round"] == round])
    averages_df = pd.DataFrame(averages).T
    print("Probability of position making regular appearance")
    print(averages_df)
    return averages_df

def forward_insight_per_round():
    forward_df = get_forwards()
    forward_df = forward_df[forward_df["GP_x"] > 30]
    key_stats = forward_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
            averages[round][stat] = (forward_df[forward_df["Round"] == round][stat] / forward_df[forward_df["Round"] == round]["Seasons"]).fillna(0).mean()
    averages_df = pd.DataFrame(averages).T
    print("Average Forward Stats For Players With Appearances")
    print(averages_df)
    return averages_df

def defender_insight_per_round():
    defender_df = get_defenders()
    defender_df = defender_df[defender_df["GP_x"] > 30]
    key_stats = defender_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        #averages[round]['GP_y'] = (defender_df[defender_df["Round"] == round]['GP_y'] / defender_df[defender_df["Round"] == round]["Seasons"]).mean()
        for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
            averages[round][stat] = (defender_df[defender_df["Round"] == round][stat] / defender_df[defender_df["Round"] == round]["Seasons"]).fillna(0).mean()
    averages_df = pd.DataFrame(averages).T
    print("Average Defender Stats For Players With Appearances")
    print(averages_df)
    return averages_df

def goalie_insight_per_round():
    goalie_df = get_goalies()
    goalie_df = goalie_df[goalie_df["GP_x"] > 10]
    goalie_df["SV%"] = goalie_df["SV"] / goalie_df["Shots"]
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        for stat in ['GS', 'T/O']:
            averages[round][stat] = (goalie_df[goalie_df["Round"] == round][stat] / goalie_df[goalie_df["Round"] == round]["Seasons"]).fillna(0).mean()
        for stat in ['GA', 'Shots', 'SV']:
            averages[round][stat] = (goalie_df[goalie_df["Round"] == round][stat] / goalie_df[goalie_df["Round"] == round]["GP_x"]).mean()
        for stat in ["SV%", "GAA"]:
            averages[round][stat] = goalie_df[goalie_df["Round"] == round][stat].mean()
        averages[round]["W/L"] = (goalie_df[goalie_df["Round"] == round]["W"] / (goalie_df[goalie_df["Round"] == round]["L"] + goalie_df[goalie_df["Round"] == round]["T/O"])).mean()
    averages_df = pd.DataFrame(averages).T
    print("Average Goalie Stats For Players With Appearances")
    print(averages_df)
    return averages_df

def scoring_player_proportion_per_round():
    forward_df = get_forwards()
    defender_df = get_defenders()
    goalie_df = get_goalies()
    
    temp_forward_df = forward_df[forward_df['GP_y'] > 0]
    temp_forward_df.loc[:, 'PPG'] = (temp_forward_df['G_y'] + temp_forward_df['A_y']) / temp_forward_df['GP_y']
    temp_defender_df = defender_df[defender_df['GP_y'] > 0]
    temp_defender_df.loc[:, 'PPG'] = (temp_defender_df['G_y'] + temp_defender_df['A_y']) / temp_defender_df['GP_y']
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        averages[round]['forwards'] = len(temp_forward_df[(temp_forward_df["PPG"] >= 0.45) & (temp_forward_df["Round"] == round)]) / len(forward_df[forward_df["Round"] == round])
        averages[round]['defenders'] = len(temp_defender_df[(temp_defender_df["PPG"] >= 0.33) & (temp_defender_df["Round"] == round)]) / len(defender_df[defender_df["Round"] == round])
    averages_df = pd.DataFrame(averages).T
    print("Good Scoring Probabilities")
    print(averages_df)
    return averages_df

#predict_forward_round()
forward_key_averages_per_round()
print()
defender_key_averages_per_round()
print()
goalie_key_averages_per_round()
print()
proportion_of_single_appearances()
print()
proportion_of_lasting_appearances()
print()
forward_insight_per_round()
print()
defender_insight_per_round()
print()
goalie_insight_per_round()
scoring_player_proportion_per_round()