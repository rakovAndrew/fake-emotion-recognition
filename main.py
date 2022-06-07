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

tensorflow.random.set_seed(13)
tensorflow.debugging.set_log_device_placement(False)
tensorflow.config.experimental.list_physical_devices('GPU')

training_stats = training_dataset_shuffled.describe()
training_stats = training_stats.transpose()
print(training_stats)

model = Sequential()
nodes = 42
model.add(Dense(nodes, input_dim=21, activation='relu'))

model.add(Dense(nodes * 4, Activation('relu')))
model.add(Dropout(0.2))
model.add(Dense(nodes * 2, Activation('relu')))
model.add(Dropout(0.2))
model.add(Dense(nodes, Activation('relu')))
model.add(Dense(nodes/2, Activation('relu')))
model.add(Dense(nodes/4, Activation('relu')))
model.add(Dense(1, Activation('sigmoid')))

learning_rate = 0.0005
optimizer = optimizers.Adam(learning_rate)
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()

EPOCHS = 200
batch_size = 13

with tensorflow.device('/CPU:0'):  # it can be with '/CPU:0'
    # with tf.device('/GPU:0'): # comment the previous line and uncomment this line to train with a GPU, if available.
    history = model.fit(
        training_dataset_data_shuffled,
        training_dataset_answers_shuffled,
        batch_size=batch_size,
        epochs=EPOCHS,
        verbose=1,
        shuffle=True,
        steps_per_epoch=int(training_dataset_data_shuffled.shape[0] / batch_size),
        validation_data=(validation_dataset_data_shuffled, validation_dataset_answers_shuffled),
    )

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
