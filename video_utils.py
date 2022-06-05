import configparser
import os.path
import time
from os.path import join

import cv2
import pandas as pd
from feat.utils import read_feat

config = configparser.ConfigParser()
config.read('config.ini')


def play_video(video):
    cap = cv2.VideoCapture(video)
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


def save_video_frames_and_aus_activity(video):
    cap = cv2.VideoCapture(video)
    csv = video.replace('video', 'csv').replace('.mp4', '.csv')
    file = pd.read_csv(csv)
    aus = file["AU06"], file["AU07"], file["AU12"], file["AU25"], file["AU26"]
    aus_info = open(os.path.join(config['Temp path']['temp directory'], 'aus_info.txt'), 'w')
    i = 0

    if not cap.isOpened():
        print("Error File Not Found")

    while cap.isOpened():
        ret, frame = cap.read()
        j = 0

        if ret == True:

            cv2.putText(img=frame, text=str(i), org=(40, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2,
                        color=(0, 255, 0), thickness=3)
            cv2.imwrite(os.path.join(config['Temp path']['temp directory'], 'frame' + str(i) + '.jpg'), frame)
            cv2.imshow('frame', frame)

            aus_info.write(str(i)+'\n')
            while j < 5:
                aus_info.write(str(aus[j][i])+'\n')
                j += 1
            print(i)

            i += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    aus_info.close()
