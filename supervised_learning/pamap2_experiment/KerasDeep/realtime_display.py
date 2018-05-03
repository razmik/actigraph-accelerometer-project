import matplotlib.animation as animation
import sys
import os
from mcfly import modelgen, find_architecture, storage

from utils import tutorial_pamap2
import matplotlib.pyplot as plt

ACTIVITIES_MAP = {
    0: 'no_activity',
    1: 'lying',
    2: 'sitting',
    3: 'standing',
    4: 'walking',
    5: 'running',
    6: 'cycling',
    7: 'nordic_walking',
    8: 'None',
    9: 'watching_tv',
    10: 'computer_work',
    11: 'car_driving',
    12: 'ascending_stairs',
    13: 'descending_stairs',
    14: 'None',
    15: 'None',
    16: 'vaccuum_cleaning',
    17: 'ironing',
    18: 'folding_laundry',
    19: 'house_cleaning',
    20: 'playing_soccer',
    21: 'None',
    22: 'None',
    23: 'None',
    24: 'rope_jumping'
}

# modelnames = ['allact_simplelr', 'allact_CNN_1', 'allact_CNN_2', 'allact_CNN_3']
modelnames = ['allact_CNN_3']
#
sys.path.insert(0, os.path.abspath('../..'))

directory_to_extract_to = 'data/full_features_allactive/'

# ## Download data and pre-proces data
data_path = tutorial_pamap2.download_preprocessed_data(directory_to_extract_to)
X_train, y_train_binary, X_val, y_val_binary, X_test, y_test_binary, labels = tutorial_pamap2.load_data(data_path)

print('x shape:', X_train.shape)
print('y shape:', y_train_binary.shape)

print('train set size:', X_train.shape[0])
print('validation set size:', X_val.shape[0])
print('test set size:', X_test.shape[0])
resultpath = os.path.join(directory_to_extract_to, 'data/models')

Accuracies = []
F1s = []
CMs = []

modelname = modelnames[0]

model_reloaded = storage.loadmodel(resultpath, modelname)

print(model_reloaded.summary())

XX = X_test
YY = y_test_binary
## Inspect model predictions on validation data
datasize = XX.shape[0]
probs = model_reloaded.predict_proba(XX[:datasize, :, :], batch_size=1)

print(labels)
# columns are predicted, rows are truth
predicted = probs.argmax(axis=1)
y_index = YY[:datasize].argmax(axis=1)

# Real time display

# y_index, predicted

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# Set up dates
test_date_times_desc = ['t-' + str(i + 1) for i in range(len(y_index))]

current_step = 1
begin_step = 0
round_count = 96  # one day equals 96 * 15 min intervals
rounds = 3  # number of days to show


def animate(i):
    global current_step
    global begin_step

    y_true = y_index[begin_step:current_step]
    y_pred = predicted[begin_step:current_step]
    x_len = range(current_step - begin_step)
    y_len = range(0, 25)

    x_axis_labels = test_date_times_desc[begin_step:current_step]

    ax1.clear()
    # ax1.scatter(x_len, y_true+0.2, s=20)
    # ax1.scatter(x_len, y_pred-0.2, s=20)
    ax1.plot(x_len, y_true + 0.15, linewidth=2)
    ax1.plot(x_len, y_pred - 0.15, linewidth=2)

    if current_step > 1:
        plt.title('Human Activity Prediction')

    plt.grid(True)
    plt.xticks(x_len, x_axis_labels, rotation=90)
    plt.yticks(y_len, list(ACTIVITIES_MAP.items()))
    plt.legend(['Actual', 'Predicted'], loc='upper left')
    plt.ylabel('Vehicle count (per 15min)')

    current_step = current_step + 1

    if current_step > (round_count * rounds):
        begin_step += 1


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
