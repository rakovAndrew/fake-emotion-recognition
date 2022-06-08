import configparser
import os

import keras.losses
import pandas as pd
import seaborn
import tensorflow
from keras import Sequential, optimizers
from keras.layers import Dense, Dropout, Activation
from keras.utils import plot_model
import matplotlib.pyplot as plt

from video_utils import get_total_video_duration_in_directory, get_video_duration, \
    parse_duration_from_seconds_to_minutes

config = configparser.ConfigParser()
config.read('config.ini')

training_dataset_path = os.path.join(config['Training path']['training directory'], 'training_dataset_file.csv')
validation_dataset_path = os.path.join(config['Validation path']['validation directory'],
                                       'validation_dataset_file.csv')

training_dataset = pd.read_csv(training_dataset_path)
training_dataset_data = training_dataset.iloc[:, :21]
training_dataset_answers = training_dataset.iloc[:, [21]]

validation_dataset = pd.read_csv(validation_dataset_path)
validation_dataset_data = validation_dataset.iloc[:, :21]
validation_dataset_answers = validation_dataset.iloc[:, [21]]

training_dataset_shuffled = training_dataset.sample(frac=1)
training_dataset_data_shuffled = training_dataset_shuffled.iloc[:, :21]
training_dataset_answers_shuffled = training_dataset_shuffled.iloc[:, [21]]

validation_dataset_shuffled = validation_dataset.sample(frac=1)
validation_dataset_data_shuffled = validation_dataset_shuffled.iloc[:, :21]
validation_dataset_answers_shuffled = validation_dataset_shuffled.iloc[:, [21]]

# tensorflow.random.set_seed(13)
# tensorflow.debugging.set_log_device_placement(False)
# tensorflow.config.experimental.list_physical_devices('GPU')

training_stats = training_dataset_shuffled.describe()
training_stats = training_stats.transpose()
# print(training_stats)


MAX_EPOCHS = 230
batch_size = 13
run = 0
runs = 6
learning_rate_border = 0.004

best_accuracy = {0.0015: 0, 0.00175: 0,
                 0.002: 0, 0.00225: 0, 0.0025: 0, 0.00275: 0,
                 0.003: 0, 0.00325: 0, 0.0035: 0, 0.00375: 0}

best_validation_accuracy = {0.0015: 0, 0.00175: 0,
                            0.002: 0, 0.00225: 0, 0.0025: 0, 0.00275: 0,
                            0.003: 0, 0.00325: 0, 0.0035: 0, 0.00375: 0}


def create_model(learning_rate):
    global model
    model = Sequential()
    nodes = 42
    model.add(Dense(nodes, input_dim=21, activation='relu'))
    model.add(Dense(nodes * 4, Activation('relu')))
    model.add(Dropout(0.2))
    model.add(Dense(nodes * 2, Activation('relu')))
    model.add(Dropout(0.2))
    model.add(Dense(nodes, Activation('relu')))
    model.add(Dense(nodes / 2, Activation('relu')))
    model.add(Dense(nodes / 4, Activation('relu')))
    model.add(Dense(1, Activation('sigmoid')))
    optimizer = optimizers.Adam(learning_rate)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])


def test_model():
    global history
    with tensorflow.device('/CPU:0'):
        history = model.fit(
            training_dataset_data_shuffled,
            training_dataset_answers_shuffled,
            batch_size=batch_size,
            epochs=epoch,
            verbose=0,
            shuffle=True,
            steps_per_epoch=int(training_dataset_data_shuffled.shape[0] / batch_size),
            validation_data=(validation_dataset_data_shuffled, validation_dataset_answers_shuffled),
        )


accuracy_by_epochs = {140: 0, 150: 0, 160: 0,
                      170: 0, 180: 0, 190: 0, 200: 0, 210: 0, 220: 0}

validation_accuracy_by_epochs = {140: 0, 150: 0, 160: 0,
                                 170: 0, 180: 0, 190: 0, 200: 0, 210: 0, 220: 0}

while run < runs:
    epoch = list(accuracy_by_epochs.keys())[0]

    while epoch < MAX_EPOCHS:
        accuracy = {}
        validation_accuracy = {}
        learning_rate = list(best_accuracy.keys())[0]

        while learning_rate < learning_rate_border:
            print('===================================')
            print('%s - %s - %.5f' % (run, epoch, learning_rate))
            print('===================================')

            create_model(learning_rate)
            # model.summary()
            test_model()

            accuracy.update({round(learning_rate, 5): history.history['accuracy'][epoch - 1]})
            validation_accuracy.update({round(learning_rate, 5): history.history['val_accuracy'][epoch - 1]})

            max_accuracy = accuracy_by_epochs.get(epoch)
            for key in accuracy.keys():
                if max_accuracy < accuracy.get(key):
                    max_accuracy = accuracy.get(key)

            accuracy_by_epochs.update({epoch: max_accuracy})

            max_validation_accuracy = validation_accuracy_by_epochs.get(epoch)
            for key in validation_accuracy.keys():
                if max_validation_accuracy < validation_accuracy.get(key):
                    max_validation_accuracy = validation_accuracy.get(key)

            validation_accuracy_by_epochs.update({epoch: max_validation_accuracy})

            learning_rate += 0.00025

        print('----------------')
        print('current accuracy')
        for key, value in accuracy.items():
            print('%.5f -- %.4f' % (key, value))
        print('----------------')

        for key, value in accuracy.items():
            if value > best_accuracy.get(key):
                best_accuracy.update({key: value})

        print('best accuracy')
        for key, value in best_accuracy.items():
            print('%.5f -- %.4f' % (key, value))
        print('----------------')

        for key, value in validation_accuracy.items():
            if value > best_validation_accuracy.get(key):
                best_validation_accuracy.update({key: value})

        print('best validation accuracy')
        for key, value in best_validation_accuracy.items():
            print('%.5f -- %.4f' % (key, value))
        print('----------------')

        epoch += 10

    run += 1

print('----------------')
print('accuracy by epochs')
for key, value in accuracy_by_epochs.items():
    print('%s -- %.4f' % (key, value))

print('----------------')
print('validation accuracy by epochs')
for key, value in validation_accuracy_by_epochs.items():
    print('%s -- %.4f' % (key, value))

print('----------------')
print('best accuracy')
for key, value in best_accuracy.items():
    print('%.5f -- %.4f' % (key, value))

print('----------------')
print('best validation accuracy')
for key, value in best_validation_accuracy.items():
    print('%.5f -- %.4f' % (key, value))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 9))
axes[0][0].plot(list(accuracy_by_epochs.keys()), list(accuracy_by_epochs.values()))
axes[0][1].plot(list(best_accuracy.keys()), list(best_accuracy.values()))
axes[1][0].plot(list(validation_accuracy_by_epochs.keys()), list(validation_accuracy_by_epochs.values()))
axes[1][1].plot(list(best_validation_accuracy.keys()), list(best_validation_accuracy.values()))
fig.tight_layout()
plt.savefig('plot')
plt.show()
# plt.plot(list(accuracy_by_epochs.keys()), list(accuracy_by_epochs.values()))
#
# plt.plot(list(best_accuracy.keys()), list(best_accuracy.values()))
# plt.show()
# while nodes > 2:
#     nodes /= 2
#     model.add(Dense(nodes, activation='relu'))
#     model.add(Dense(nodes, activation='relu'))
#     model.add(Dropout(0.2))
#     print(nodes)
#
# # model.add(Dense(16, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(4, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(4, activation='relu'))
# # model.add(Dropout(0.2))
# nodes /= 2
# model.add(Dense(nodes, activation='sigmoid'))
# print(nodes)
# # plot_model(model, show_shapes=True, show_layer_names=True)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(training_dataset_data_shuffled, training_dataset_answers_shuffled, epochs=100, batch_size=10, verbose=0)
# loss, accuracy = model.evaluate(validation_dataset_data_shuffled, validation_dataset_answers_shuffled, verbose=0)
# print('Model Loss: %.2f, Accuracy: %.2f' % ((loss * 100), (accuracy * 100)))
#
# predictions = model.predict(validation_dataset_data_shuffled)
# m = 0
# for i in range(len(validation_dataset_data_shuffled)):
#     if predictions[i] == 1:
#         m += 1
#     print('Predicted %d---> Expected %d' % (predictions[i], validation_dataset_answers_shuffled ['truthfulness'].iloc[i]))
# print(m)
