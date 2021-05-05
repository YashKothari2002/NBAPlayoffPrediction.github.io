import pandas as pd
from os import listdir
import numpy as np

years = []
teams = pd.read_csv('teams.csv')
games = pd.read_csv('games.csv')

for file in sorted(listdir('teamdata')):
    years.append((int(file.split('stats')[0]), pd.read_csv('teamdata/' + file)))

# convert team names into index
for _, year in years:
    teamnames = []
    ids = []
    for team in year['Team'].values:
        name = team.split(' ')
        name = name[len(name) - 1]
        if len(teams[teams['NICKNAME'].str.contains(name)]) != 0:
            id = teams[teams['NICKNAME'].str.contains(name)]['TEAM_ID'].values[0]
            ids.append(id)
            teamnames.append(team)
    year['Team'] = year['Team'].replace(teamnames, ids)
    teamnames.clear()

# construct matching
data = []  # (team_home, team_away, winner 0 = away 1 = home)
for i in range(len(games.values)):
    home = games['HOME_TEAM_ID'][i]
    visitor = games['VISITOR_TEAM_ID'][i]
    winner = games['HOME_TEAM_WINS'][i]
    date = games['GAME_DATE_EST'][i]
    data.append((home, visitor, winner, date))

ids = []


def insert_columns(d: pd.DataFrame, label='HOME_'):
    for key in years[0][1].keys():
        d.insert(len(d), label + key, 0)


def create_data(team_id, date: str, label='HOME_'):
    dat = {}
    y, m, d = date.split('-')
    y, m, d = int(y), int(m), int(d)

    if m > 4:
        y += 1
    y -= 1
    tmp = [item[1] for item in years if item[0] == y]
    if len(tmp) == 0:
        return None
    teams_data = tmp[0]
    if len(teams_data[teams_data['Team'] == team_id]) == 0:
        return None

    for key in teams_data.keys():
        col = label + key
        dat[col] = teams_data[teams_data['Team'] == team_id][key].values[0]
    return dat


df = pd.DataFrame()

# setup dataframe
insert_columns(df, 'HOME_')
insert_columns(df, 'AWAY_')
df.insert(len(df), 'WINNER', 0)
df.insert(len(df), 'DATE', 0)

for i in range(len(data)):
    h, a, w, date = data[i]
    home = create_data(h, date)
    away = create_data(a, date, label='AWAY_')
    if home is not None and away is not None:
        d = home
        d.update(away)
        d['WINNER'] = w
        y, m, _ = date.split('-')
        y, m = int(y), int(m)

        if m > 4:
            y += 1
        d['DATE'] = y
        df.loc[i] = d
        if i % 100 == 0:
            print('done ', i, '/', len(data), str(data[i]))

df.to_csv('data.csv')
