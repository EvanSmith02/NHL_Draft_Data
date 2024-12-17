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

def raw_draft_data_to_file():
    drafts_by_year = {}
    for year in range(2007,2022):
        url = "https://www.eliteprospects.com/draft/nhl-entry-draft/" + str(year)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        filename = './data/raw/draft_year_' + str(year) + ".html"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(soup.prettify())
        #drafts_by_year[year] = read_prospects(soup, year)
    #return drafts_by_year

def raw_advanced_data_to_file():
    for year in range(2015, 2025):
        print(year)
        url = "https://www.hockey-reference.com/leagues/NHL_" + str(year) + "_goalies.html"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        filename = './data/raw/goalie_stat_year_' + str(year) + ".html"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(soup.prettify())
        
        url = "https://www.hockey-reference.com/leagues/NHL_" + str(year) + "_skaters.html"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        filename = './data/raw/skater_stat_year_' + str(year) + ".html"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(soup.prettify())

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
    draft_data = pd.DataFrame(columns=["Round", "Draft Position", "Player", "Position", "Team", "Seasons", "GP", "G", "A", "TP", "PIM", "Year"])
    current_round = 1
    for round in data:
        for player in round:
            print(player[3])
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
                            "Year": year
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
                            "Year": year
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
                            "Year": year
                        }
                draft_data.loc[len(draft_data)] = draft_entry
            except:
                print(f"Error in {player[3]}'s format")
        current_round += 1
    return draft_data

raw_draft_data_to_file()
raw_advanced_data_to_file()