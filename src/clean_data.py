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

def write_cleaned_draft_data():
    drafts_by_year = get_drafted_players()
    players_df = pd.DataFrame(columns=["Round", "Draft Position", "Player", "Position", "Team", "Seasons", "GP", "G", "A", "TP", "PIM", "Year", "PPS"])
    for i in drafts_by_year.keys():
        for j in range(0, len(drafts_by_year[i])):
            current_row = drafts_by_year[i].iloc[j]
            players_df.loc[len(players_df)] = current_row
    #with open('data/processed/draft_data.csv', 'w', encoding='utf-8') as file:
    players_df.to_csv('data/processed/draft_data.csv', encoding='utf-8')
            

def get_drafted_players():
    drafts_by_year = {}
    for year in range(2010,2022):
        with open('./data/raw/draft_year_' + str(year) + ".html", 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            drafts_by_year[year] = read_prospects(soup, year)
    return drafts_by_year

def get_goalie_stats():
    hr_goalie_df = pd.DataFrame(columns = ['Rk', 'Player', 'Age', 'Team', 'Pos', 'GP', 'GS', 'W', 'L', 'T/O', 'GA', 'Shots', 'SV', 'SV%', 'GAA', 'SO', 'MIN', 'QS', 'QS%', 'RBS', 'GA%-', 'GSAA', 'GAA/A', 'GPS', 'G', 'A', 'PTS', 'PIM', 'Awards'])
    for year in range(2015, 2025):
        with open('./data/raw/goalie_stat_year_' + str(year) + ".html", 'r', encoding='utf-8') as file:
            tables = pd.read_html(file)
            print(f"Found {len(tables)} tables")
            tables[0].columns = ['Rk', 'Player', 'Age', 'Team', 'Pos', 'GP', 'GS', 'W', 'L', 'T/O', 'GA', 'Shots', 'SV', 'SV%', 'GAA', 'SO', 'MIN', 'QS', 'QS%', 'RBS', 'GA%-', 'GSAA', 'GAA/A', 'GPS', 'G', 'A', 'PTS', 'PIM', 'Awards']
            for i in range(len(tables[0])):
                row = tables[0].loc[i]
                if row["Player"] in hr_goalie_df["Player"].values:
                    for col in ['GP', 'GS', 'W', 'L', 'T/O', 'GA', 'Shots', 'SV', 'SO', 'MIN', 'QS']:
                        hr_goalie_df.loc[hr_goalie_df["Player"] == row["Player"], col] += row[col]
                    for col in ['SV%', 'QS%']:
                        hr_goalie_df.loc[hr_goalie_df["Player"] == row["Player"], col] = max(hr_goalie_df[hr_goalie_df["Player"] == row["Player"]].loc[:, col].iloc[0], row[col])
                else:
                    hr_goalie_df.loc[len(hr_goalie_df)] = row
    hr_goalie_df.to_csv('data/processed/goalie_stats.csv', encoding='utf-8')
    
def get_skater_stats():
    hr_skater_df = pd.DataFrame(columns = ['Rk', 'Player', 'Age', 'Team', 'Pos', 'GP', 'G', 'A', 'PTS', '+/-',
       'PIM', 'EVG', 'PPG', 'SHG', 'GWG', 'EV', 'PP', 'SH', 'SOG', 'SPCT',
       'TSA', 'TOI', 'ATOI', 'FOW', 'FOL', 'FO%', 'BLK', 'HIT', 'TAKE', 'GIVE'])
    url = "https://www.hockey-reference.com/leagues/NHL_2024_skaters.html"
    for year in range(2015, 2025):
        with open('./data/raw/skater_stat_year_' + str(year) + ".html", 'r', encoding='utf-8') as file:
            tables = pd.read_html(file)
            print(f"Found {len(tables)} tables")
            tables[0].columns = ['Rk', 'Player', 'Age', 'Team', 'Pos', 'GP', 'G', 'A', 'PTS', '+/-',
                                'PIM', 'EVG', 'PPG', 'SHG', 'GWG', 'EV', 'PP', 'SH', 'SOG', 'SPCT',
                                'TSA', 'TOI', 'ATOI', 'FOW', 'FOL', 'FO%', 'BLK', 'HIT', 'TAKE', 'GIVE', 'Awards']
            tables[0].drop(["Awards"], axis = 1)
            for i in range(len(tables[0])):
                row = tables[0].loc[i]
                #print(row)
                if row["Player"] in hr_skater_df["Player"].values:
                    for col in ['GP', 'G', 'A', 'PTS', '+/-', 'PIM', 'EVG', 'PPG', 'SHG', 'GWG', 'EV', 'PP', 'SH', 'SOG', 'BLK', 'HIT', 'TAKE', 'GIVE']:
                        hr_skater_df.loc[hr_skater_df["Player"] == row["Player"], col] += row[col]
                    hr_skater_df.loc[hr_skater_df["Player"] == row["Player"], "ATOI"] = max(hr_skater_df[hr_skater_df["Player"] == row["Player"]].loc[:, "ATOI"].iloc[0], row["ATOI"])
                else:
                    hr_skater_df.loc[len(hr_skater_df)] = row
    hr_skater_df.to_csv('data/processed/skater_stats.csv', encoding='utf-8')
    
def merge_skater():
    draft_df = pd.read_csv('data/processed/draft_data.csv')
    skater_df = pd.read_csv('data/processed/skater_stats.csv')
    goalie_df = pd.read_csv('data/processed/goalie_stats.csv')
    
    merged_df = pd.merge(skater_df, draft_df[draft_df["Position"] != 'G'], on='Player', how='right')
    merged_df = merged_df.drop(columns = ['Pos', 'Unnamed: 0_x'])
    merged_df = merged_df.fillna(0)
    forward_df = merged_df[merged_df['Position'] == 'F']
    defender_df = merged_df[merged_df['Position'] == 'D']
    merge_goalies = pd.merge(goalie_df, draft_df[draft_df["Position"] == 'G'], on = 'Player', how='right').fillna(0)
    
    forward_df.to_csv('data/processed/forward_merged_stats.csv', encoding='utf-8')
    defender_df.to_csv('data/processed/defender_merged_stats.csv', encoding='utf-8')
    merge_goalies.to_csv('data/processed/goalie_merged_stats.csv', encoding='utf-8')

def read_prospects(soup, year):
    outer_table = soup.find('div', id = 'drafted-players')
    table_div = outer_table.find('div', class_=lambda x: x and 'table-wizard' in x)
    table = table_div.find('table')
    rows = table.find_all('tr')[1:]
    data = []
    current_round = 0
    for row in rows:
        cols = row.find_all('td')
        if len(cols) == 1:
            current_round = int(re.findall(r'\d', cols[0].text.strip())[0])
            data.append([])
        elif cols:
            row_data = [col.text.strip() for col in cols]
            data[current_round - 1].append(row_data)
    draft_data = pd.DataFrame(columns=["Round", "Draft Position", "Player", "Position", "Team", "Seasons", "GP", "G", "A", "TP", "PIM", "Year", "PPS"])
    current_round = 1
    for round in data:
        for player in round:
            #print(player[3])
            try: # The 'Tyson Hinds' error
                if len(re.findall(r'\((.)\)', player[3])) > 0: # Fixing edge case from teams like the coyotes not following the rules
                    if player[6] == '-': # For Goalies
                        draft_entry = {
                            "Round": current_round,
                            "Draft Position": int(re.findall(r'\d+', player[0])[0]),
                            "Team": player[2],
                            "Player": player[3].split(" (")[0],
                            "Position": re.findall(r'\((.)\)', player[3])[0],
                            "Seasons": int(player[4]),
                            "GP": int(player[5]),
                            "G": 0,
                            "A": 0,
                            "TP": 0,
                            "PIM": 0,
                            "Year": year,
                            "PPS": 0
                        }
                    elif player[5]:
                        draft_entry = {
                            "Round": current_round,
                            "Draft Position": int(re.findall(r'\d+', player[0])[0]),
                            "Team": player[2],
                            "Player": player[3].split(" (")[0],
                            "Position": re.findall(r'\((.)\)', player[3])[0],
                            "Seasons": int(player[4]),
                            "GP": int(player[5]),
                            "G": int(player[6]),
                            "A": int(player[7]),
                            "TP": int(player[8]),
                            "PIM": int(player[9]),
                            "Year": year,
                            "PPS": (int(player[6]) + int(player[7])) / int(player[4])
                        }
                    else: # For Players who have zero appearences
                        draft_entry = {
                            "Round": current_round,
                            "Draft Position": int(re.findall(r'\d+', player[0])[0]),
                            "Team": player[2],
                            "Player": player[3].split(" (")[0],
                            "Position": re.findall(r'\((.)\)', player[3])[0],
                            "Seasons": 0,
                            "GP": 0,
                            "G": 0,
                            "A": 0,
                            "TP": 0,
                            "PIM": 0,
                            "Year": year,
                            "PPS": 0
                        }
                draft_data.loc[len(draft_data)] = draft_entry
            except:
                print(f"Error in {player[3]}'s format")
        current_round += 1
    return draft_data

write_cleaned_draft_data()
get_goalie_stats()
get_skater_stats()
merge_skater()