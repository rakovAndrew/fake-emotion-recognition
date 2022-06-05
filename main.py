import configparser
import os
from os.path import join

import numpy as np
import pandas as pd
from feat import Fex
from feat.utils import read_feat

import video_utils
from csv_data_utils import join_csv_files, join_csv_files_in_directory, extract_and_save_csv_data_from_video, \
    remove_duplicated_row, remove_rows_with_duplicated_frames_from_all_videos, \
    compress_all_csv_data_and_save_csv_with_frames, save_emotion_truthfulness

config = configparser.ConfigParser()
config.read('config.ini')

true_training_dataset_path = os.path.join(config['Training path']['training directory'],
                                          'true_training_dataset_file.csv')
fake_training_dataset_path = os.path.join(config['Training path']['training directory'],
                                          'fake_training_dataset_file.csv')
true_validation_dataset_path = os.path.join(config['Validation path']['validation directory'],
                                            'true_validation_dataset_file.csv')
fake_validation_dataset_path = os.path.join(config['Validation path']['validation directory'],
                                            'fake_validation_dataset_file.csv')

true_training_dataset_file = join_csv_files_in_directory(
    config['Training path']['true pleasure compressed csv directory'],
    true_training_dataset_path)
save_emotion_truthfulness(true_training_dataset_path, true_training_dataset_path, 1)

fake_training_dataset_file = join_csv_files_in_directory(
    config['Training path']['fake pleasure compressed csv directory'],
    fake_training_dataset_path)
save_emotion_truthfulness(fake_training_dataset_path, fake_training_dataset_path, 0)

true_validation_dataset_file = join_csv_files_in_directory(
    config['Validation path']['true pleasure compressed csv directory'],
    true_validation_dataset_path)
save_emotion_truthfulness(true_validation_dataset_path, true_validation_dataset_path, 1)

fake_validation_dataset_file = join_csv_files_in_directory(
    config['Validation path']['fake pleasure compressed csv directory'],
    fake_validation_dataset_path)
save_emotion_truthfulness(fake_validation_dataset_path, fake_validation_dataset_path, 0)

print(true_training_dataset_file)
print(fake_training_dataset_file)
print(true_validation_dataset_file)
print(fake_validation_dataset_file)

training_dataset_file = join_csv_files(
    os.path.join(config['Training path']['training directory'], 'training_dataset_file.csv'),
    true_training_dataset_path,
    fake_training_dataset_path)
validation_dataset_file = join_csv_files(
    os.path.join(config['Validation path']['validation directory'], 'validation_dataset_file.csv'),
    true_validation_dataset_path,
    fake_validation_dataset_path)

training_dataset = {}

print(training_dataset_file)
print(validation_dataset_file)
