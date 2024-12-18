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
import analyze_data

def category_by_round():
    career_player, callup_player, bust_player = {}, {}, {}
    players_by_round = pd.read_csv('data/processed/draft_data.csv')
    for round in players_by_round['Round'].unique():
        career_player[round] = (len(players_by_round[(players_by_round["GP"] >= 100) & (players_by_round["Round"] == round)]))
        callup_player[round] = (len(players_by_round[(players_by_round["GP"] >= 20) & (players_by_round["GP"] < 100) & (players_by_round["Round"] == round)]))
        bust_player[round] = (len(players_by_round[(players_by_round["GP"] < 20) & (players_by_round["Round"] == round)]))

    bar_width = 0.15
    x = np.arange(1, len(players_by_round['Round'].unique()) + 1)
    x_career = x
    x_callup = x + bar_width
    x_bust = x + 2 * bar_width
    plt.bar(x_career, career_player.values(), width = 0.15, label='x >= 100', color='blue')
    plt.bar(x_callup, callup_player.values(), width = 0.15, label='25 <= x < 100', color='orange')
    plt.bar(x_bust, bust_player.values(), width = 0.15, label='x < 25', color='green')

    plt.xlabel('Round')
    plt.ylabel('Player Count')
    plt.title('Success Ratio Per Round')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def stats_by_pick():
    forward_df = analyze_data.get_forwards()
    key_stats = forward_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    fig, axes = plt.subplots(2, 5, figsize=(14, 14))
    axes = axes.flatten()
    for i, stat in enumerate(key_stats):
        axes[i].scatter(forward_df['Draft Position'], forward_df[stat] / forward_df["GP_y"])
        axes[i].set_title(f'{stat} vs Pick Number')
        axes[i].set_xlabel('Pick Number')
        axes[i].set_ylabel(stat)

    plt.tight_layout()
    plt.show()
    
    defender_df = analyze_data.get_defenders()
    key_stats = defender_df[['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']]
    fig, axes = plt.subplots(2, 5, figsize=(14, 14))
    axes = axes.flatten()
    for i, stat in enumerate(key_stats):
        axes[i].scatter(defender_df['Draft Position'], defender_df[stat] / defender_df["GP_y"])
        axes[i].set_title(f'{stat} vs Pick Number')
        axes[i].set_xlabel('Pick Number')
        axes[i].set_ylabel(stat)

    plt.tight_layout()
    plt.show()

def averages_by_round():
    forward_df =  analyze_data.forward_key_averages_per_round()
    fig, axes = plt.subplots(2, 5, figsize=(14, 13))
    axes = axes.flatten()
    for i, stat in enumerate(forward_df.columns):
        axes[i].scatter(range(1,8), forward_df[stat])
        axes[i].set_title(f'{stat} vs Draft Round')
        axes[i].set_xlabel('Draft Round')
        axes[i].set_ylabel(stat)
    plt.tight_layout()
    plt.show()
    
    defender_df =  analyze_data.defender_insight_per_round()
    fig, axes = plt.subplots(2, 5, figsize=(14, 13))
    axes = axes.flatten()
    for i, stat in enumerate(defender_df.columns):
        axes[i].scatter(range(1,8), defender_df[stat])
        axes[i].set_title(f'{stat} vs Draft Round')
        axes[i].set_xlabel('Draft Round')
        axes[i].set_ylabel(stat)

    plt.tight_layout()
    plt.show()
    
    goalie_df =  analyze_data.goalie_insight_per_round()
    fig, axes = plt.subplots(2, 5, figsize=(14, 13))
    axes = axes.flatten()
    for i, stat in enumerate(goalie_df.columns):
        axes[i].scatter(range(1,8), goalie_df[stat])
        axes[i].set_title(f'{stat} vs Draft Round')
        axes[i].set_xlabel('Draft Round')
        axes[i].set_ylabel(stat)

    plt.tight_layout()
    plt.show()

def percentage_success_by_position():
    data = analyze_data.proportion_of_single_appearances()
    plt.figure(figsize=(10, 6))
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)
    plt.xlabel('Round')
    plt.ylabel('Percent Appear')
    plt.title('Success Ratio Per Round')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    data = analyze_data.proportion_of_lasting_appearances()
    plt.figure(figsize=(10, 6))
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)
    plt.xlabel('Round')
    plt.ylabel('Percent Success')
    plt.title('Success Ratio Per Round (Skater > 42gp and Goalie > 15)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def percentage_scoring_success_by_position():
    data = analyze_data.scoring_player_proportion_per_round()
    plt.figure(figsize=(10, 6))
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)
    plt.xlabel('Round')
    plt.ylabel('Percent Appear')
    plt.title('Percent Becoming Decent Scorers (F > 0.45 PPG & D > 0.30 PPG)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def averages_by_pick():
    forward_df =  analyze_data.get_forward_mean_for_pick()
    defender_df = analyze_data.get_defender_mean_for_pick()
    f_draft_positions, f_regression_line = analyze_data.predict_forward_pick()
    print()
    d_draft_positions, d_regression_line = analyze_data.predict_defender_pick()
    
    fig, axes = plt.subplots(2, 5, figsize=(14, 13))
    axes = axes.flatten()
    for i, stat in enumerate(forward_df.columns):
        axes[i].scatter(range(1,225), forward_df[stat])
        axes[i].scatter(f_draft_positions, f_regression_line[stat], color='red')
        #axes[i].plot(draft_positions, regression_line[stat], color='red', label='Regression Line')
        axes[i].set_title(f'{stat} vs Pick #')
        axes[i].set_xlabel('Pick #')
        axes[i].set_ylabel(stat)
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(2, 5, figsize=(14, 13))
    axes = axes.flatten()
    for i, stat in enumerate(defender_df.columns):
        axes[i].scatter(range(1,225), defender_df[stat])
        axes[i].scatter(d_draft_positions, d_regression_line[stat], color='red')
        #axes[i].plot(draft_positions, regression_line[stat], color='red', label='Regression Line')
        axes[i].set_title(f'{stat} vs Pick #')
        axes[i].set_xlabel('Pick #')
        axes[i].set_ylabel(stat)
    plt.tight_layout()
    plt.show()

def averages_by_height():
    forward_df =  analyze_data.forward_key_averages_per_height()
    fig, axes = plt.subplots(2, 5, figsize=(14, 13))
    axes = axes.flatten()
    for i, stat in enumerate(forward_df.columns):
        axes[i].scatter(forward_df.index.to_list(), forward_df[stat])
        axes[i].set_title(f'{stat} vs Height')
        axes[i].set_xlabel('Inches')
        axes[i].set_ylabel(stat)
    plt.tight_layout()
    plt.show()
    
    defender_df =  analyze_data.defender_key_averages_per_height()
    fig, axes = plt.subplots(2, 5, figsize=(14, 13))
    axes = axes.flatten()
    for i, stat in enumerate(defender_df.columns):
        axes[i].scatter(defender_df.index.to_list(), defender_df[stat])
        axes[i].set_title(f'{stat} vs Height')
        axes[i].set_xlabel('Inches')
        axes[i].set_ylabel(stat)

    plt.tight_layout()
    plt.show()
    
    goalie_df =  analyze_data.goalie_key_averages_per_height()
    fig, axes = plt.subplots(2, 5, figsize=(14, 13))
    axes = axes.flatten()
    for i, stat in enumerate(goalie_df.columns):
        axes[i].scatter(goalie_df.index.to_list(), goalie_df[stat])
        axes[i].set_title(f'{stat} vs Height')
        axes[i].set_xlabel('Inches')
        axes[i].set_ylabel(stat)

    plt.tight_layout()
    plt.show()

def predicted_stats_height_weight():
    op_heights, op_weights = analyze_data.predict_forward_stat_height_weight()
    stats = ['GP_y', 'G_y', 'A_y', 'TP', 'PIM_y', 'BLK', 'HIT', 'TAKE', 'GIVE', '+/-']
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    axes = axes.flatten()
    print(len(op_heights))
    axes[0].scatter(stats, op_heights)
    axes[0].set_title("Predicted Optimal Forward Heights For Each Stat")
    axes[0].set_xlabel('Stats')
    axes[0].set_ylabel('Inches')
    axes[1].scatter(stats, op_weights)
    axes[1].set_title("Predicted Optimal Forward Weights For Each Stat")
    axes[1].set_xlabel('Stats')
    axes[1].set_ylabel('Pounds')    
    op_heights, op_weights = analyze_data.predict_defender_stat_height_weight()
    axes[2].scatter(stats, op_heights)
    axes[2].set_title("Predicted Optimal Defender Heights For Each Stat")
    axes[2].set_xlabel('Stats')
    axes[2].set_ylabel('Inches')
    axes[3].scatter(stats, op_weights)
    axes[3].set_title("Predicted Optimal Defender Weights For Each Stat")
    axes[3].set_xlabel('Stats')
    axes[3].set_ylabel('Pounds')
    plt.tight_layout()
    plt.show()

predicted_stats_height_weight()
percentage_scoring_success_by_position()
category_by_round()
#stats_by_pick()
averages_by_round()

averages_by_pick()
averages_by_height()
percentage_success_by_position()