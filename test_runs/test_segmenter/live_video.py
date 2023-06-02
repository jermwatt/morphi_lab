import time
import cv2
from morphi_lab.segmenter import Segmenter
import fire


def test_live_video(conf=0.5):
    video = cv2.VideoCapture(0)
    time.sleep(2)
    if (video.isOpened() is False):
        print("Error reading video file")

    # read until end of video
    while True:
        # read frame
        ret, frame = video.read()

        # segment frame
        segmenter = Segmenter(conf=conf)
        segmenter.read_img(frame)
        segmenter.segment()
        segmenter.plot_segmentations()

        if ret is True:
            # Display the resulting frame
            cv2.imshow('Frame', segmenter.img)

            # break condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    fire.Fire(test_live_video)
