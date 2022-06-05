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
    compress_all_csv_data_and_save_csv_with_frames

config = configparser.ConfigParser()
config.read('config.ini')

remove_rows_with_duplicated_frames_from_all_videos()
compress_all_csv_data_and_save_csv_with_frames()
true_training_dataset_file = join_csv_files_in_directory(config['Training path']['true pleasure compressed csv directory'])
fake_training_dataset_file = join_csv_files_in_directory(config['Training path']['fake pleasure compressed csv directory'])

true_validation_dataset_file = join_csv_files_in_directory(config['Validation path']['true pleasure compressed csv directory'])
fake_validation_dataset_file = join_csv_files_in_directory(config['Validation path']['fake pleasure compressed csv directory'])

print(true_training_dataset_file)
print(fake_training_dataset_file)
print(true_validation_dataset_file)
print(fake_validation_dataset_file)

# training_dataset_file = join_csv_files(true_training_dataset_file, fake_training_dataset_file)
# validation_dataset_file = join_csv_files(true_validation_dataset_file, fake_validation_dataset_file)
#
# training_dataset = {}
#
# print(training_dataset_file)
# print(validation_dataset_file)
#
