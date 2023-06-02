import time
import cv2


def test_live_video():
    video = cv2.VideoCapture(0)
    time.sleep(2)
    if (video.isOpened() is False):
        print("Error reading video file")

    # read until end of video
    while True:
        # read frame
        ret, frame = video.read()

        if ret is True:
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # break condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_live_video()
