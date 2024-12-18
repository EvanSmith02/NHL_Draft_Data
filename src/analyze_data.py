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

def predict_forward_stat_height_weight():
    forward_df = get_forwards()
    poly = pre.PolynomialFeatures(degree=2)
    for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
        forward_df[stat] = (forward_df[stat] / forward_df["Seasons"])
        forward_df[stat] = forward_df[stat].replace([np.inf, -np.inf], np.nan).fillna(0)
    forward_df = forward_df[forward_df['GP_y'] > 0]
    key_stats = forward_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    
    y = forward_df[["Height", "Weight"]]
    height_weight_grid = np.array([[h, w] for h in forward_df[forward_df["Height"] > 0]['Height'] for w in forward_df[forward_df["Weight"] > 0]['Weight']])
    model = lm.LinearRegression()
    y_preds = {}
    optimal_heights = []
    optimal_weights = []
    for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
        X_poly = poly.fit_transform(y)
        model.fit(X_poly, key_stats[stat])
        y_preds[stat] = model.predict(X_poly)
        max_index = np.argmax(y_preds[stat])
        optimal_height, optimal_weight = height_weight_grid[max_index]
        print(f"Optimal Height for {stat} is {optimal_height}")
        print(f"Optimal Weight for {stat} is {optimal_weight}")
        optimal_heights.append(optimal_height)
        optimal_weights.append(optimal_weight)
        mae = skm.mean_absolute_error(key_stats[stat], y_preds[stat])
        print(f"Forward Mean Absolute Error For {stat} = {mae}")
    return optimal_heights, optimal_weights

def predict_defender_stat_height_weight():
    defender_df = get_defenders()
    poly = pre.PolynomialFeatures(degree=2)
    for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
        defender_df[stat] = (defender_df[stat] / defender_df["Seasons"])
        defender_df[stat] = defender_df[stat].replace([np.inf, -np.inf], np.nan).fillna(0)
    defender_df = defender_df[defender_df['GP_y'] > 0]
    key_stats = defender_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    
    y = defender_df[["Height", "Weight"]]
    height_weight_grid = np.array([[h, w] for h in defender_df[defender_df["Height"] > 0]['Height'] for w in defender_df[defender_df["Weight"] > 0]['Weight']])
    model = lm.LinearRegression()
    y_preds = {}
    optimal_heights = []
    optimal_weights = []
    for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
        X_poly = poly.fit_transform(y)
        model.fit(X_poly, key_stats[stat])
        y_preds[stat] = model.predict(X_poly)
        max_index = np.argmax(y_preds[stat])
        optimal_height, optimal_weight = height_weight_grid[max_index]
        print(f"Optimal Height for {stat} is {optimal_height}")
        print(f"Optimal Weight for {stat} is {optimal_weight}")
        optimal_heights.append(optimal_height)
        optimal_weights.append(optimal_weight)
        mae = skm.mean_absolute_error(key_stats[stat], y_preds[stat])
        print(f"Forward Mean Absolute Error For {stat} = {mae}")
    return optimal_heights, optimal_weights

def predict_forward_pick():
    forward_df = get_forwards()
    
    poly = pre.PolynomialFeatures(degree=2)
    
    for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
        forward_df[stat] = (forward_df[stat] / forward_df["Seasons"])
        forward_df[stat] = forward_df[stat].replace([np.inf, -np.inf], np.nan).fillna(0)
    key_stats = forward_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    
    y = forward_df["Draft Position"]
    model = lm.LinearRegression()
    y_preds = {}
    for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
        X_poly = poly.fit_transform(y.values.reshape(-1, 1))  # Transform to polynomial features
        model.fit(X_poly, key_stats[stat])  # Fit the polynomial regression model
        y_preds[stat] = model.predict(X_poly)
        mae = skm.mean_absolute_error(key_stats[stat], y_preds[stat])
        print(f"Forward Mean Absolute Error For {stat} = {mae}")
    return y, y_preds

def predict_defender_pick():
    defender_df = get_defenders()
    
    poly = pre.PolynomialFeatures(degree=2)
    
    for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
        defender_df[stat] = (defender_df[stat] / defender_df["Seasons"])
        defender_df[stat] = defender_df[stat].replace([np.inf, -np.inf], np.nan).fillna(0)
    key_stats = defender_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    
    y = defender_df["Draft Position"]
    model = lm.LinearRegression()
    y_preds = {}
    for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
        X_poly = poly.fit_transform(y.values.reshape(-1, 1))  # Transform to polynomial features
        model.fit(X_poly, key_stats[stat])  # Fit the polynomial regression model
        y_preds[stat] = model.predict(X_poly)
        mae = skm.mean_absolute_error(key_stats[stat], y_preds[stat])
        print(f"Defender Mean Absolute Error For {stat} = {mae}")
    return y, y_preds

def predict_goalie_picks():
    goalies_df = get_goalies()
    
    poly = pre.PolynomialFeatures(degree=2)
    for stat in ['GS', 'W', 'L', 'T/O']:
        goalies_df[stat] = (goalies_df[stat] / goalies_df["Seasons"])
        goalies_df[stat] = goalies_df[stat].replace([np.inf, -np.inf], np.nan).fillna(0)
    for stat in ['GA', 'Shots', 'SV']:
        goalies_df[stat] = (goalies_df[stat] / goalies_df["GP_x"])
        goalies_df[stat] = goalies_df[stat].replace([np.inf, -np.inf], np.nan).fillna(0)
    key_stats = goalies_df[['GS', 'W', 'L', 'T/O', 'GA', 'Shots', 'SV', 'SV%' ,'GAA']]
    y = goalies_df["Draft Position"]
    model = lm.LinearRegression()
    y_preds = {}
    for stat in ['GS', 'W', 'L', 'T/O', 'GA', 'Shots', 'SV', 'SV%' ,'GAA']:
        X_poly = poly.fit_transform(y.values.reshape(-1, 1))  # Transform to polynomial features
        model.fit(X_poly, key_stats[stat])  # Fit the polynomial regression model
        y_preds[stat] = model.predict(X_poly)
        mae = skm.mean_absolute_error(key_stats[stat], y_preds[stat])
        print(f"Goalie Mean Absolute Error For {stat} = {mae}")
    return y, y_preds

def correlation_analysis():
    forward_df = get_forwards()[['Round', 'Draft Position', 'GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    print(f"Forward Correlations = \n{forward_df.corr()}")
    defender_df = get_defenders()[['Round', 'Draft Position', 'GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    print(f"Defender Correlations = \n{defender_df.corr()}")
    goalie_df = get_goalies()[['Round', 'Draft Position', 'GS', 'W', 'L', 'T/O', 'GA', 'Shots', 'SV', 'SV%' ,'GAA']]
    print(f"Goalie Correlations = \n{goalie_df.corr()}")
    
def per_season_correlation_analysis():
    forward_df = get_forwards()[['Round', 'Draft Position', 'GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-', 'Seasons', 'Height', 'Weight']]
    for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
        forward_df[stat] = (forward_df[stat] / forward_df["Seasons"])
        forward_df[stat] = forward_df[stat].replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"Forward Season Average Correlations = \n{forward_df.corr()}")
    defender_df = get_defenders()[['Round', 'Draft Position', 'GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-', 'Seasons', 'Height', 'Weight']]
    for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
        defender_df[stat] = (defender_df[stat] / defender_df["Seasons"])
        defender_df[stat] = defender_df[stat].replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"Defender Season Average Correlations = \n{defender_df.corr()}")
    goalie_df = get_goalies()[['Round', 'Draft Position', 'GS', 'W', 'L', 'T/O', 'GA', 'Shots', 'SV', 'SV%' ,'GAA', 'Seasons', 'GP_x', 'Height', 'Weight']]
    for stat in ['GS', 'W', 'L', 'T/O']:
        goalie_df[stat] = (goalie_df[stat] / goalie_df["Seasons"])
        goalie_df[stat] = goalie_df[stat].replace([np.inf, -np.inf], np.nan).fillna(0)
    for stat in ['GA', 'Shots', 'SV']:
        goalie_df[stat] = (goalie_df[stat] / goalie_df["GP_x"])
        goalie_df[stat] = goalie_df[stat].replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"Goalie Season Average Correlations = \n{goalie_df.corr()}")
    
def forward_key_averages_per_round():
    forward_df = get_forwards()
    forward_df = forward_df[forward_df["GP_y"] != 0]
    key_stats = forward_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
            averages[round][stat] = (forward_df[forward_df["Round"] == round][stat] / forward_df[forward_df["Round"] == round]["Seasons"]).fillna(0).mean()
    averages_df = pd.DataFrame(averages).T
    print("Forward Averages Per Round")
    print(averages_df)
    return averages_df
    
def defender_key_averages_per_round():
    defender_df = get_defenders()
    defender_df = defender_df[defender_df["GP_y"] != 0]
    key_stats = defender_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        #averages[round]['GP_y'] = (defender_df[defender_df["Round"] == round]['GP_y'] / defender_df[defender_df["Round"] == round]["Seasons"]).mean()
        for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
            defender_df[stat] = defender_df[stat].replace([np.inf, -np.inf], np.nan).fillna(0)
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
        for stat in ['GA', 'Shots', 'SV']:
            averages[round][stat] = (goalie_df[goalie_df["Round"] == round][stat] / goalie_df[goalie_df["Round"] == round]["GP_x"]).mean()
        for stat in ['SV%' ,'GAA']:
            averages[round][stat] = (goalie_df[goalie_df["Round"] == round][stat]).mean()
    averages_df = pd.DataFrame(averages).T
    print("Goalie Averages per round")
    print(averages_df)
    return averages

def forward_key_averages_per_height():
    forward_df = get_forwards()
    key_stats = forward_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    averages = {}
    forward_df = forward_df[forward_df['Height'] > 0]
    for inch in forward_df['Height'].unique():
        averages[inch] = {}
        for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
            averages[inch][stat] = (forward_df[forward_df["Height"] == inch][stat] / forward_df[forward_df["Height"] == inch]["Seasons"]).fillna(0).mean()
    averages_df = pd.DataFrame(averages).T
    print("Forward Averages Per Height")
    print(averages_df)
    return averages_df

def defender_key_averages_per_height():
    defender_df = get_forwards()
    key_stats = defender_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    defender_df = defender_df[defender_df['Height'] > 0]
    averages = {}
    for inch in defender_df['Height'].unique():
        averages[inch] = {}
        for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
            averages[inch][stat] = (defender_df[defender_df["Height"] == inch][stat] / defender_df[defender_df["Height"] == inch]["Seasons"]).fillna(0).mean()
    averages_df = pd.DataFrame(averages).T
    print("Defender Averages Per Height")
    print(averages_df)
    return averages_df

def goalie_key_averages_per_height():
    goalie_df = get_goalies()
    goalie_df = goalie_df[goalie_df["GP_x"] != 0]
    goalie_df = goalie_df[goalie_df['Height'] > 0]
    averages = {}
    for inch in goalie_df['Height'].unique():
        averages[inch] = {}
        for stat in ['GS', 'W', 'L', 'T/O']:
            averages[inch][stat] = (goalie_df[goalie_df["Height"] == inch][stat] / goalie_df[goalie_df["Height"] == inch]["Seasons"]).fillna(0).mean()
        for stat in ['GA', 'Shots', 'SV']:
            averages[inch][stat] = (goalie_df[goalie_df["Height"] == inch][stat] / goalie_df[goalie_df["Height"] == inch]["GP_x"]).mean()
        for stat in ['SV%' ,'GAA']:
            averages[inch][stat] = (goalie_df[goalie_df["Height"] == inch][stat]).mean()
    averages_df = pd.DataFrame(averages).T
    print("Goalie Averages per Height")
    print(averages_df)
    return averages_df

def forward_height_and_weight_per_round():
    forward_df = get_forwards()
    key_stats = forward_df[['Height', 'Weight']]
    forward_df = forward_df[forward_df['Height'] > 0]
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        for stat in ['Height', 'Weight']:
            averages[round][stat] = (forward_df[forward_df["Round"] == round][stat]).fillna(0).mean()
    averages_df = pd.DataFrame(averages).T
    print("Forward Average Height Per Round")
    print(averages_df)
    return averages_df

def defender_height_and_weight_per_round():
    defender_df = get_defenders()
    key_stats = defender_df[['Height', 'Weight']]
    defender_df = defender_df[defender_df['Height'] > 0]
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        for stat in ['Height', 'Weight']:
            averages[round][stat] = (defender_df[defender_df["Round"] == round][stat]).fillna(0).mean()
    averages_df = pd.DataFrame(averages).T
    print("Defender Average Height Per Round")
    print(averages_df)
    return averages_df

def goalie_height_and_weight_per_round():
    goalie_df = get_goalies()
    goalie_df = goalie_df[goalie_df['Height'] > 0]
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        for stat in ['Height', 'Weight']:
            averages[round][stat] = (goalie_df[goalie_df["Round"] == round][stat]).fillna(0).mean()
    averages_df = pd.DataFrame(averages).T
    print("Goalie Average Height Per Round")
    print(averages_df)
    return averages_df

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
    return averages_df

def proportion_of_lasting_appearances():
    forward_df = get_forwards()
    defender_df = get_defenders()
    goalie_df = get_goalies()
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        averages[round]['forwards'] = len(forward_df[(forward_df["GP_y"] > 42) & (forward_df["Round"] == round)]) / len(forward_df[forward_df["Round"] == round])
        averages[round]['defenders'] = len(defender_df[(defender_df["GP_y"] > 42) & (defender_df["Round"] == round)]) / len(defender_df[defender_df["Round"] == round])
        averages[round]['goalie'] = len(goalie_df[(goalie_df["GP_y"] > 15) & (goalie_df["Round"] == round)]) / len(goalie_df[goalie_df["Round"] == round])
    averages_df = pd.DataFrame(averages).T
    print("Probability of position making regular appearance")
    print(averages_df)
    return averages_df

def forward_insight_per_round():
    forward_df = get_forwards()
    forward_df = forward_df[forward_df["GP_y"] > 30]
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
    defender_df = defender_df[defender_df["GP_y"] > 30]
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
    goalie_df = goalie_df[goalie_df["GP_y"] > 10]
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
    
    temp_forward_df = forward_df#[forward_df['GP_y'] > 0]
    temp_forward_df.loc[:, 'PPG'] = ((temp_forward_df['G_y'] + temp_forward_df['A_y']) / temp_forward_df['GP_y']).fillna(0)
    temp_defender_df = defender_df#[defender_df['GP_y'] > 0]
    temp_defender_df.loc[:, 'PPG'] = (temp_defender_df['G_y'] + temp_defender_df['A_y']) / temp_defender_df['GP_y'].fillna(0)
    averages = {1:{}, 2: {}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}}
    for round in averages.keys():
        averages[round]['forwards'] = len(temp_forward_df[(temp_forward_df["PPG"] >= 0.45) & (temp_forward_df["Round"] == round)]) / len(forward_df[forward_df["Round"] == round])
        averages[round]['defenders'] = len(temp_defender_df[(temp_defender_df["PPG"] >= 0.30) & (temp_defender_df["Round"] == round)]) / len(defender_df[defender_df["Round"] == round])
    averages_df = pd.DataFrame(averages).T
    print("Good Scoring Probabilities")
    print(averages_df)
    return averages_df

def get_forward_mean_for_pick():
    forward_df = get_forwards()
    key_stats = forward_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    averages = {}
    for i in range(1,225):
        averages[i] = {}
    for pick in range(1,225):
        for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
            averages[pick][stat] = (forward_df[forward_df["Draft Position"] == pick][stat] / forward_df[forward_df["Draft Position"] == pick]["Seasons"]).fillna(0).mean()
    averages_df = pd.DataFrame(averages).T
    print("Average Forward Stats Per Pick #")
    print(averages_df)
    return averages_df.fillna(0)

def get_defender_mean_for_pick():
    defender_df = get_defenders()
    key_stats = defender_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    averages = {}
    for i in range(1,225):
        averages[i] = {}
    for pick in range(1,225):
        for stat in ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']:
            averages[pick][stat] = (defender_df[defender_df["Draft Position"] == pick][stat] / defender_df[defender_df["Draft Position"] == pick]["Seasons"]).fillna(0).mean()
    averages_df = pd.DataFrame(averages).T
    print("Average Forward Stats Per Pick #")
    print(averages_df)
    return averages_df.fillna(0)




proportion_of_single_appearances()
print()
proportion_of_lasting_appearances()
print()
predict_forward_pick()
print()
predict_defender_pick()
print()
predict_goalie_picks()
print()
forward_key_averages_per_round()
print()
defender_key_averages_per_round()
print()
goalie_key_averages_per_round()
print()

forward_insight_per_round()
print()
defender_insight_per_round()
print()
goalie_insight_per_round()
scoring_player_proportion_per_round()

#get_forward_mean_for_pick()
forward_key_averages_per_height()
print()
defender_key_averages_per_height()
print()
goalie_key_averages_per_height()
print()
forward_height_and_weight_per_round()
print()
defender_height_and_weight_per_round()
print()
goalie_height_and_weight_per_round()
print()
correlation_analysis()
print()
per_season_correlation_analysis()