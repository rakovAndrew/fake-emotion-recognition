import configparser
import os
from os.path import join

import numpy as np
from feat import Fex
from feat.plotting import draw_muscles, plot_face
from feat.utils import read_feat

import video_utils
from csv_data_utils import remove_specific_row, save_specific_column, find_mean_by_columns, \
    find_and_save_emotion_duration

cols = ['frame', 'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU11', 'AU12', 'AU14', 'AU15', 'AU17',
        'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU28', 'AU43']
cols_without_frames = ['mean_AU01', 'mean_AU02', 'mean_AU04', 'mean_AU05', 'mean_AU06', 'mean_AU07', 'mean_AU09',
                       'mean_AU10', 'mean_AU11', 'mean_AU12', 'mean_AU14',
                       'mean_AU15', 'mean_AU17', 'mean_AU20', 'mean_AU23', 'mean_AU24', 'mean_AU25', 'mean_AU26',
                       'mean_AU28', 'mean_AU43', 'mean_duration']

dict = {
    'AU06': 0.1953,
    'AU07': 0.3844,
    'AU12': 0.2541,
    'AU25': 0.2298,
    'AU26': 0.1793
}

# config = configparser.ConfigParser()
# config.read('config.ini')
#
# for file in os.listdir(config['Paths']['true pleasure csv directory']):
#     csv_directory = os.path.join(config['Paths']['true pleasure csv directory'], file)
#     compressed_csv_directory = os.path.join(config['Paths']['true pleasure compressed csv directory'], file)
#     compressed_csv_with_frames_directory = os.path.join(
#         config['Paths']['true pleasure compressed csv with frames directory'], file)
#
#     remove_specific_row(csv_directory,
#                         compressed_csv_directory,
#                         dict)
#     save_specific_column(compressed_csv_directory,
#                          compressed_csv_directory,
#                          cols)
#     find_and_save_emotion_duration(compressed_csv_directory,
#                                    compressed_csv_with_frames_directory)
#     find_mean_by_columns(compressed_csv_with_frames_directory,
#                          compressed_csv_directory)
#     save_specific_column(compressed_csv_directory,
#                          compressed_csv_directory,
#                          cols_without_frames)


video_utils.save_video_frames_and_aus_activity('dataset/training/video/true/pleasure/002454-HD-GusNonnii-P1043170.mp4')