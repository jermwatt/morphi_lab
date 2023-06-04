import cv2
from ultralytics import YOLO


def main():
    # import model
    model = YOLO('/models/YOLOv8s-seg.pt')

    # act and track
    for result in model.track(source=0, show=True, stream=True):
        frame = result.orig_img
        cv2.imshow('yolo8', frame)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
