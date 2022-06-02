import configparser
import shutil

from feat.detector import Detector
import os
import cv2
import time


def method_name():
    cap = cv2.VideoCapture('C141_Trim.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened():
        print("Error File Not Found")
    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:

            time.sleep(1 / fps)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()


face_model = "retinaface"
landmark_model = "pfld"
au_model = "jaanet"
emotion_model = "fer"
detector = Detector(
    face_model=face_model,
    landmark_model=landmark_model,
    emotion_model=emotion_model
)

config = configparser.ConfigParser()
config.read('config.ini')

for file in os.listdir(config['Paths']['fake pleasure video directory']):
    new_file_name = os.path.splitext(file)[0] + '.csv'
    video_prediction = detector.detect_video(os.path.join(config['Paths']['fake pleasure video directory'], file), outputFname=new_file_name)

    shutil.move(new_file_name, os.path.join(config['Paths']['fake pleasure csv directory'], new_file_name))

# method_name()
