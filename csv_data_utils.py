import configparser
import glob
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

cols = ['frame', 'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU11', 'AU12', 'AU14', 'AU15', 'AU17',
        'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU28', 'AU43']
cols_without_frames = ['mean_AU01', 'mean_AU02', 'mean_AU04', 'mean_AU05', 'mean_AU06', 'mean_AU07', 'mean_AU09',
                       'mean_AU10', 'mean_AU11', 'mean_AU12', 'mean_AU14',
                       'mean_AU15', 'mean_AU17', 'mean_AU20', 'mean_AU23', 'mean_AU24', 'mean_AU25', 'mean_AU26',
                       'mean_AU28', 'mean_AU43', 'mean_duration']

happiness_aus = {
    'AU06': 0.1953,
    'AU07': 0.3844,
    'AU12': 0.2541,
    'AU25': 0.2298,
    'AU26': 0.1793
}


def extract_and_save_csv_data_from_video(video, video_directory, csv_directory, skip_frames=1):
    new_file_name = video.replace('.mp4', '.csv')
    video_prediction = detector.detect_video(os.path.join(video_directory, video), skip_frames=skip_frames,
                                             outputFname=new_file_name)
    shutil.move(new_file_name,
                os.path.join(csv_directory, new_file_name))
    return video_prediction


def extract_csv_data_from_all_videos(video_directory, csv_directory):
    for file in os.listdir(video_directory):
        extract_and_save_csv_data_from_video(file,
                                             video_directory,
                                             csv_directory)


def extract_csv_data_from_all_training_fake_videos(
        fake_pleasure_video_directory=config['Training path']['fake pleasure video directory'],
        fake_pleasure_csv_directory=config['Training path']['fake pleasure csv directory']):
    extract_csv_data_from_all_videos(fake_pleasure_video_directory,
                                     fake_pleasure_csv_directory)


def extract_csv_data_from_all_training_true_videos(
        true_pleasure_video_directory=config['Training path']['true pleasure video directory'],
        true_pleasure_csv_directory=config['Training path']['true pleasure csv directory']):
    extract_csv_data_from_all_videos(true_pleasure_video_directory,
                                     true_pleasure_csv_directory)


def extract_csv_data_from_all_validation_fake_videos(
        fake_pleasure_video_directory=config['Validation path']['fake pleasure video directory'],
        fake_pleasure_csv_directory=config['Validation path']['fake pleasure csv directory']):
    extract_csv_data_from_all_videos(fake_pleasure_video_directory,
                                     fake_pleasure_csv_directory)


def extract_csv_data_from_all_validation_true_videos(
        true_pleasure_video_directory=config['Validation path']['true pleasure video directory'],
        true_pleasure_csv_directory=config['Validation path']['true pleasure csv directory']):
    extract_csv_data_from_all_videos(true_pleasure_video_directory,
                                     true_pleasure_csv_directory)


def remove_specific_row(file, new_file, columns_with_values):
    csv_file = pd.read_csv(file)
    for key, value in columns_with_values.items():
        csv_file = csv_file[eval("csv_file.{}".format(key)) >= value]
    csv_file.to_csv(new_file, index=False)


def remove_duplicated_row(file, new_file, columns):
    csv_file = pd.read_csv(file)
    csv_file = csv_file.drop_duplicates(subset=columns, keep='first')
    csv_file.to_csv(new_file, index=False)


def remove_rows_with_duplicated_frames(file, new_file):
    remove_duplicated_row(file, new_file, ['frame'])


def remove_rows_with_duplicated_frames_from_all_videos():
    paths = [config['Validation path']['true pleasure csv directory'],
             config['Validation path']['fake pleasure csv directory'],
             config['Training path']['true pleasure csv directory'],
             config['Training path']['fake pleasure csv directory']]

    for path in paths:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            remove_duplicated_row(file_path, file_path, ['frame'])


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


def compress_csv_data_and_save_csv_with_frames(
        pleasure_csv_directory,
        pleasure_compressed_csv_directory,
        pleasure_compressed_csv_with_frames_directory):
    for file in os.listdir(pleasure_csv_directory):
        csv_directory = os.path.join(pleasure_csv_directory, file)
        compressed_csv_directory = os.path.join(pleasure_compressed_csv_directory,
                                                file)
        compressed_csv_with_frames_directory = os.path.join(
            pleasure_compressed_csv_with_frames_directory, file)

        remove_specific_row(csv_directory,
                            compressed_csv_directory,
                            happiness_aus)
        save_specific_column(compressed_csv_directory,
                             compressed_csv_directory,
                             cols)
        find_and_save_emotion_duration(compressed_csv_directory,
                                       compressed_csv_with_frames_directory)
        find_mean_by_columns(compressed_csv_with_frames_directory,
                             compressed_csv_directory)
        save_specific_column(compressed_csv_directory,
                             compressed_csv_directory,
                             cols_without_frames)


def compress_validation_true_pleasure_csv_data_and_save_csv_with_frames():
    compress_csv_data_and_save_csv_with_frames(
        config['Validation path']['true pleasure csv directory'],
        config['Validation path']['true pleasure compressed csv directory'],
        config['Validation path']['true pleasure compressed csv with frames directory']
    )


def compress_validation_fake_pleasure_csv_data_and_save_csv_with_frames():
    compress_csv_data_and_save_csv_with_frames(
        config['Validation path']['fake pleasure csv directory'],
        config['Validation path']['fake pleasure compressed csv directory'],
        config['Validation path']['fake pleasure compressed csv with frames directory']
    )


def compress_training_true_pleasure_csv_data_and_save_csv_with_frames():
    compress_csv_data_and_save_csv_with_frames(
        config['Training path']['true pleasure csv directory'],
        config['Training path']['true pleasure compressed csv directory'],
        config['Training path']['true pleasure compressed csv with frames directory']
    )


def compress_training_fake_pleasure_csv_data_and_save_csv_with_frames():
    compress_csv_data_and_save_csv_with_frames(
        config['Training path']['fake pleasure csv directory'],
        config['Training path']['fake pleasure compressed csv directory'],
        config['Training path']['fake pleasure compressed csv with frames directory']
    )


def compress_all_csv_data_and_save_csv_with_frames():
    compress_validation_true_pleasure_csv_data_and_save_csv_with_frames()
    compress_validation_fake_pleasure_csv_data_and_save_csv_with_frames()
    compress_training_true_pleasure_csv_data_and_save_csv_with_frames()
    compress_training_fake_pleasure_csv_data_and_save_csv_with_frames()


def join_csv_files_in_directory(files_directory):
    joined_files = os.path.join('%s' % files_directory, '*.csv')
    joined_list = glob.glob(joined_files)
    fat_file = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
    return fat_file


def join_csv_files(*files):
    joined_list = []
    for file in files:
        joined_list.append(file)
    fat_file = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
    return fat_file
