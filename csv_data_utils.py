import configparser
import os
import shutil

import pandas as pd
from feat import Detector
from feat.utils import read_feat

face_model = "retinaface"
landmark_model = "pfld"
emotion_model = "fer"
detector = Detector(
    face_model=face_model,
    landmark_model=landmark_model,
    emotion_model=emotion_model
)

config = configparser.ConfigParser()
config.read('config.ini')


def extract_and_save_csv_data_from_video(video, video_directory, csv_directory):
    new_file_name = video.replace('.mp4', '.csv')
    video_prediction = detector.detect_video(os.path.join(video_directory, video),
                                             outputFname=new_file_name)
    shutil.move(new_file_name,
                os.path.join(csv_directory, new_file_name))
    return video_prediction


def extract_csv_data_from_all_videos(video_directory, csv_directory):
    for file in os.listdir(video_directory):
        extract_and_save_csv_data_from_video(file,
                                             video_directory,
                                             csv_directory)


def extract_csv_data_from_all_fake_videos(
        fake_pleasure_video_directory=config['Paths']['fake pleasure video directory'],
        fake_pleasure_csv_directory=config['Paths']['fake pleasure csv directory']):
    extract_csv_data_from_all_videos(fake_pleasure_video_directory,
                                     fake_pleasure_csv_directory)


def extract_csv_data_from_all_true_videos(
        true_pleasure_video_directory=config['Paths']['true pleasure video directory'],
        true_pleasure_csv_directory=config['Paths']['true pleasure csv directory']):
    extract_csv_data_from_all_videos(true_pleasure_video_directory,
                                     true_pleasure_csv_directory)


def remove_specific_row(file, new_file, columns_with_values):
    csv_file = pd.read_csv(file)
    for key, value in columns_with_values.items():
        csv_file = csv_file[eval("csv_file.{}".format(key)) >= value]
    csv_file.to_csv(new_file, index=False)


def save_specific_column(file, new_file, columns):
    csv_file = pd.read_csv(file)
    csv_file = csv_file[columns]
    csv_file.to_csv(new_file, index=False)


def find_mean_by_columns(file, new_file):
    feat = read_feat(file)
    feat = feat.extract_mean()
    feat.to_csv(new_file, index=False)


def find_and_save_emotion_duration(file, new_file):
    csv_file = pd.read_csv(file)
    new_column = csv_file['AU43'] + 1
    csv_file['duration'] = new_column
    csv_file['duration'] = csv_file.shape[0] / 30
    csv_file.to_csv(new_file, index=False)
