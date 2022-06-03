import time

import cv2


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
