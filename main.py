import configparser
import os

import pandas as pd
import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.utils import plot_model

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

training_dataset_data_shuffled = training_dataset_data.sample(frac=1)
training_dataset_answers_shuffled = training_dataset_answers.sample(frac=1)

validation_dataset_data_shuffled = validation_dataset_data.sample(frac=1)
validation_dataset_answers_shuffled = validation_dataset_answers.sample(frac=1)

model = Sequential()
model.add_module(Dense(32, input_dim=29, activation='relu'))
model.add_module(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add_module(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add_module(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add_module(Dense(1, activation='sigmoid'))
plot_model(model, show_shapes=True, show_layer_names=True)
# model.compile(loss)
