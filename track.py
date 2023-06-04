import cv2
from ultralytics import YOLO
import os
current_dir = os.getcwd()


def main():
    # import model
    model = YOLO(current_dir + '/models/YOLOv8n-seg.pt')

    # act and track
    for result in model.track(source=0,
                              show=True,
                              stream=True,
                              imgsz=(480)):
        frame = result.orig_img
        cv2.imshow('yolo8', frame)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
