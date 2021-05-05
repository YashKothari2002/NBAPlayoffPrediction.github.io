import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython.display import clear_output
import seaborn as sns


sns_colors = sns.color_palette('colorblind')


# Load dataset.
data = pd.read_csv('data.csv')
size = len(data)

dftrain = data[data['DATE'] < 2018]
dfeval2018 = data[data['DATE'] == 2018]
dfeval2019 = data[data['DATE'] == 2019]
dfeval2020 = data[data['DATE'] == 2020]
dfevalall = data[data['DATE'] >= 2018]
y_train = dftrain.pop('WINNER')
y2018_eval = dfeval2018.pop('WINNER')
y2019_eval = dfeval2019.pop('WINNER')
y2020_eval = dfeval2020.pop('WINNER')
yall_eval = dfevalall.pop('WINNER')

dftrain.pop('DATE')
fc = tf.feature_column
NUMERIC_COLUMNS = dftrain.keys().values

feature_columns = []
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(fc.numeric_column(feature_name,
                                             dtype=tf.float32))

# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)


def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = (dataset
                   .repeat(n_epochs)
                   .batch(NUM_EXAMPLES))
        return dataset

    return input_fn


# Training and evaluation input functions.
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn2018 = make_input_fn(dfeval2018, y2018_eval, shuffle=False, n_epochs=1)
eval_input_fn2019 = make_input_fn(dfeval2019, y2019_eval, shuffle=False, n_epochs=1)
eval_input_fn2020 = make_input_fn(dfeval2020, y2020_eval, shuffle=False, n_epochs=1)
eval_input_fnall = make_input_fn(dfevalall, yall_eval, shuffle=False, n_epochs=1)

params = {
    'n_trees': 50,
    'max_depth': 3,
    'n_batches_per_layer': 1,
    # You must enable center_bias = True to get DFCs. This will force the model to
    # make an initial prediction before using any features (e.g. use the mean of
    # the training labels for regression or log odds for classification when
    # using cross entropy loss).
    'center_bias': True
}

est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
# Train model.
est.train(train_input_fn, max_steps=100)

# Evaluation.
results = est.evaluate(eval_input_fn2018)
pd.Series(results).to_frame()

in_memory_params = dict(params)
in_memory_params['n_batches_per_layer'] = 1


# In-memory input_fn does not use batching.
def make_inmemory_train_input_fn(X, y):
    y = np.expand_dims(y, axis=1)

    def input_fn():
        return dict(X), y

    return input_fn


train_input_fn = make_inmemory_train_input_fn(dftrain, y_train)

# Train the model.
est = tf.estimator.BoostedTreesClassifier(
    feature_columns,
    train_in_memory=True,
    **in_memory_params)

est.train(train_input_fn)
clear_output()


# Boilerplate code for plotting :)
def _get_color(value):
    """To make positive DFCs plot green, negative DFCs plot red."""
    green, red = sns.color_palette()[2:4]
    if value >= 0: return green
    return red


def _add_feature_values(feature_values, ax):
    """Display feature's values on left of plot."""
    x_coord = ax.get_xlim()[0]
    OFFSET = 0.15
    for y_coord, (feat_name, feat_val) in enumerate(feature_values.items()):
        t = plt.text(x_coord, y_coord - OFFSET, '{}'.format(feat_val), size=12)
        t.set_bbox(dict(facecolor='white', alpha=0.5))
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_weight('bold')
    t = plt.text(x_coord, y_coord + 1 - OFFSET, 'feature\nvalue',
                 fontproperties=font, size=12)


def plot_example(example, dfeval, id):
    TOP_N = 8  # View top 8 features.
    sorted_ix = example.abs().sort_values()[-TOP_N:].index  # Sort by magnitude.
    example = example[sorted_ix]
    colors = example.map(_get_color).tolist()
    ax = example.to_frame().plot(kind='barh',
                                 color=[colors],
                                 legend=None,
                                 alpha=0.75,
                                 figsize=(10, 6))
    ax.grid(False, axis='y')
    ax.set_yticklabels(ax.get_yticklabels(), size=14)

    # Add feature values.
    _add_feature_values(dfeval.iloc[id][sorted_ix], ax)
    return ax


# Boilerplate plotting code.
def dist_violin_plot(df_dfc, ID, dfeval):
    # Initialize plot.
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create example dataframe.
    TOP_N = 8  # View top 8 features.
    example = df_dfc.iloc[ID]
    ix = example.abs().sort_values()[-TOP_N:].index
    example = example[ix]
    example_df = example.to_frame(name='dfc')

    # Add contributions of entire distribution.
    parts = ax.violinplot([df_dfc[w] for w in ix],
                          vert=False,
                          showextrema=False,
                          widths=0.7,
                          positions=np.arange(len(ix)))
    face_color = sns_colors[0]
    alpha = 0.15
    for pc in parts['bodies']:
        pc.set_facecolor(face_color)
        pc.set_alpha(alpha)

    # Add feature values.
    _add_feature_values(dfeval.iloc[ID][ix], ax)

    # Add local contributions.
    ax.scatter(example,
               np.arange(example.shape[0]),
               color=sns.color_palette()[2],
               s=100,
               marker="s",
               label='contributions for example')

    # Legend
    # Proxy plot, to show violinplot dist on legend.
    ax.plot([0, 0], [1, 1], label='eval set contributions\ndistributions',
            color=face_color, alpha=alpha, linewidth=10)
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large',
                       frameon=True)
    legend.get_frame().set_facecolor('white')

    # Format plot.
    ax.set_yticks(np.arange(example.shape[0]))
    ax.set_yticklabels(example.index)
    ax.grid(False, axis='y')
    ax.set_xlabel('Contribution to predicted probability', size=14)


def run(label, fn, dfeval, yeval):
    results = est.predict(fn)
    pred_dicts = list(est.experimental_predict_with_explanations(fn))
    labels = yeval.values
    probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
    df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
    df_dfc.describe().T
    dist_violin_plot(df_dfc, 100, dfeval)

    i = 0
    homes, aways, winners, predictions = [], [], [], []
    for result in results:
        homes.append(dfeval['HOME_Team'][dfeval['HOME_Team'].index[i]])
        aways.append(dfeval['AWAY_Team'][dfeval['AWAY_Team'].index[i]])
        winners.append(yeval[yeval.index[i]])
        predictions.append(result['class_ids'][0])
        i += 1
    data = {'Home': homes, 'Away': aways, 'Winner': winners, 'Prediction': predictions}
    df = pd.DataFrame(data)
    df.to_csv('b2_results/predictions' + label + '.csv')


run('2018', eval_input_fn2018, dfeval2018, y2018_eval)
run('2019', eval_input_fn2019, dfeval2019, y2019_eval)
run('2020', eval_input_fn2020, dfeval2020, y2020_eval)
results = est.evaluate(eval_input_fnall)
data = [['Gradient Boost', results['accuracy'], results['precision'], results['average_loss'], results['recall']]]
df = pd.DataFrame(data, columns=['Method', 'Accuracy', 'Precision', 'Loss', 'Recall'])
df.to_csv('out/all.csv')
df.to_html('out/all.html')
print(results)
