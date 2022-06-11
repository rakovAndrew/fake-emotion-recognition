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
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV

from csv_data_utils import extract_csv_data_from_all_training_true_videos, \
    extract_csv_data_from_all_training_fake_videos, compress_all_csv_data_and_save_csv_with_frames, \
    merge_all_csv_files_into_training_and_validation_files
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

def fill_dictionary_using_border_values_and_step(left_border, right_border, step):
    dictionary = {}
    while left_border < right_border:
        dictionary.update({round(left_border, 5): 0})
        left_border += step

    return dictionary


MAX_EPOCHS = 270
MIN_EPOCHS = 100
EPOCHS_STEP = 20

batch_size = 10
run = 0
runs = 4

MAX_LEARNING_RATE = 0.02
MIN_LEARNING_RATE = 0.001
LEARNING_RATE_STEP = 0.0015

best_accuracy = fill_dictionary_using_border_values_and_step(MIN_LEARNING_RATE, MAX_LEARNING_RATE, LEARNING_RATE_STEP)

best_validation_accuracy = fill_dictionary_using_border_values_and_step(MIN_LEARNING_RATE, MAX_LEARNING_RATE,
                                                                        LEARNING_RATE_STEP)


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
    return model


def test_model(epoch):
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


accuracy_by_epochs = fill_dictionary_using_border_values_and_step(MIN_EPOCHS, MAX_EPOCHS, EPOCHS_STEP)

validation_accuracy_by_epochs = fill_dictionary_using_border_values_and_step(MIN_EPOCHS, MAX_EPOCHS, EPOCHS_STEP)

while run < runs:
    epoch = list(accuracy_by_epochs.keys())[0]

    while epoch < MAX_EPOCHS:
        accuracy = {}
        validation_accuracy = {}
        learning_rate = list(best_accuracy.keys())[0]

        while learning_rate < MAX_LEARNING_RATE:
            print('===================================')
            print('%s - %s - %.5f' % (run, epoch, learning_rate))
            print('===================================')

            create_model(learning_rate)
            # model.summary()
            test_model(epoch)

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

            learning_rate += LEARNING_RATE_STEP

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

        epoch += EPOCHS_STEP

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


def tune_and_create_mlp(learning_rate=0.00004, epoch=280):
    model = create_model(learning_rate=learning_rate)
    model.summary()
    test_model(epoch=epoch)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Cross-Validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Cross-Validation'], loc='upper left')
    plt.show()

    return model


# tune_and_create_mlp()


def tune_and_train_svm():
    ml = svm.SVC()
    param_grid = {'C': [1, 10, 100, 1000, 10000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}
    grid = GridSearchCV(ml, param_grid, refit=True, verbose=1, cv=15)
    grid_search = grid.fit(training_dataset_data_shuffled, training_dataset_answers_shuffled)
    print(grid_search.best_params_)
    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))
    y_test_hat = grid.predict(validation_dataset_data_shuffled)
    test_accuracy = accuracy_score(validation_dataset_answers_shuffled, y_test_hat) * 100
    print("Accuracy for our testing dataset with tuning is : {:.2f}%".format(test_accuracy))
    confusion_matrix(validation_dataset_answers_shuffled, y_test_hat)
    disp = plot_confusion_matrix(grid, validation_dataset_data_shuffled, validation_dataset_answers_shuffled,
                                 cmap=plt.cm.Blues)
    plt.show()


# tune_and_train_svm()


# plt.plot(list(accuracy_by_epochs.keys()), list(accuracy_by_epochs.values()))
#
# plt.plot(list(best_accuracy.keys()), list(best_accuracy.values()))
# plt.show()


# # plot_model(model, show_shapes=True, show_layer_names=True)

def check_model_prediction():
    predictions = model.predict(validation_dataset_data_shuffled)
    m = 0
    for i in range(len(validation_dataset_data_shuffled)):
        if predictions[i] == 1:
            m += 1
        print('Predicted %d---> Expected %d' % (
        predictions[i], validation_dataset_answers_shuffled['truthfulness'].iloc[i]))
    print(m)


# check_model_prediction()
