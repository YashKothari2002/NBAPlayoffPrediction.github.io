import pandas as pd
from scipy.stats import kendalltau

teams = pd.read_csv('teams.csv')

keytoname = lambda key: teams[teams['TEAM_ID'] == key]['ABBREVIATION'].values[0] + ' ' + \
                        teams[teams['TEAM_ID'] == key]['NICKNAME'].values[0]


def add(d: dict, key):
    name = keytoname(key)
    if name not in d.keys():
        d[name] = 1
    else:
        d[name] += 1


def ranked(year, df: pd.DataFrame, dir):
    df['Percent Win'] = df['Predicted Wins'] / df['Games Played']
    p = df.sort_values(by=['Percent Win'], ascending=False, ignore_index=True)
    p.index += 1
    p.to_html(dir + '/' + year + "_predicted_ranked.html")
    p.to_csv(dir + '/' + year + "_predicted_ranked.csv")

    df['Percent Win'] = df['True Wins'] / df['Games Played']
    t = df.sort_values(by=['Percent Win'], ascending=False, ignore_index=True)
    t.index += 1
    t.to_html(dir + '/' + year + "_true_ranked.html")
    t.to_csv(dir + '/' + year + "_true_ranked.csv")

    print(year)
    print('------------')
    print(kendalltau(t['Team'], p['Team']))
    print('\n')


def pretty(year, dir='b2_results', sort=False):
    wins = {}
    predicted_wins = {}
    played = {}
    csv = pd.read_csv(dir + '/predictions' + year + '.csv')
    for i in range(len(csv.values)):
        away = csv['Away'][i]
        home = csv['Home'][i]
        add(played, away)
        add(played, home)
        if csv['Winner'][i] == 0:
            key = csv['Away'][i]
            add(wins, key)
        else:
            key = csv['Home'][i]
            add(wins, key)
        if csv['Prediction'][i] == 0:
            key = csv['Away'][i]
            add(predicted_wins, key)
        else:
            key = csv['Home'][i]
            add(predicted_wins, key)
    data = []
    for key in wins.keys():
        data.append([key, predicted_wins[key] if key in predicted_wins.keys() else 0, wins[key], played[key],
                     str(round(100 *
                               ((predicted_wins[key] if key in predicted_wins.keys() else 0) - wins[key]) / wins[key],
                               1)) + '%'])
    df = pd.DataFrame(data, columns=['Team', 'Predicted Wins', 'True Wins', 'Games Played', 'Error'])
    df.index += 1
    if not sort:
        df.to_html(dir + '/' + year + '_final.html')
        df.to_csv(dir + '/' + year + '_final.csv')
    else:
        ranked(year, df, dir)


pretty('2018', dir='b2_results', sort=True)
pretty('2019', dir='b2_results', sort=True)
pretty('2020', dir='b2_results', sort=True)
