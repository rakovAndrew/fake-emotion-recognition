import configparser
import os
import shutil

from feat import Detector


face_model = "retinaface"
landmark_model = "psafld"
emotion_model = "fer"
detector = Detector(
    face_model=face_model,
    landmark_model=landmark_model,
    emotion_model=emotion_model
)

config = configparser.ConfigParser()


def extract_and_save_csv_data_from_video(video, video_directory, csv_directory):
    new_file_name = os.path.splitext(video)[0] + '.csv'
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


def extract_csv_data_from_all_fake_videos(fake_pleasure_video_directory=config['Paths']['fake pleasure video directory'],
                                          fake_pleasure_csv_directory=config['Paths']['fake pleasure csv directory']):
    extract_csv_data_from_all_videos(fake_pleasure_video_directory,
                                     fake_pleasure_csv_directory)


def extract_csv_data_from_all_true_videos(true_pleasure_video_directory=config['Paths']['true pleasure video directory'],
                                          true_pleasure_csv_directory=config['Paths']['true pleasure csv directory']):
    extract_csv_data_from_all_videos(true_pleasure_video_directory,
                                     true_pleasure_csv_directory)
