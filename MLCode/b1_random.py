import pandas as pd
import random
import numpy as np


data = pd.read_csv('data.csv')

dfeval2018 = data[data['DATE'] == 2018]
dfeval2019 = data[data['DATE'] == 2019]
dfeval2020 = data[data['DATE'] == 2020]


def run(dfeval: pd.DataFrame, label=''):
    homes, aways, winners, predictions = [], [], [], []
    for i in range(len(dfeval.values)):
        homes.append(dfeval['HOME_Team'][dfeval['HOME_Team'].index[i]])
        aways.append(dfeval['AWAY_Team'][dfeval['AWAY_Team'].index[i]])
        winners.append(dfeval['WINNER'][dfeval['WINNER'].index[i]])
        predictions.append(round(random.random()))
    data = {'Home': homes, 'Away': aways, 'Winner': winners, 'Prediction': predictions}
    df = pd.DataFrame(data)
    df.to_csv('random_results/predictions' + label + '.csv')

    w = np.array(winners)
    p = np.array(predictions)
    correct = (np.sum(np.logical_not(np.logical_xor(w, p))))
    print(correct / len(w))


run(dfeval2018, '2018')
run(dfeval2019, '2019')
run(dfeval2020, '2020')
