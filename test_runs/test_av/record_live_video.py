import time
import cv2
import fire


def test_live_video(output_path):
    video = cv2.VideoCapture(0)
    time.sleep(2)
    if (video.isOpened() is False):
        print("Error reading video file")

    # get frame width and height from img
    success, img = video.read()

    # get frame width and height from img 
    frame_width = img.shape[1]
    frame_height = img.shape[0]

    # define codec and create VideoWriter object
    print(output_path)
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             30, size)

    # read until end of video
    while True:
        # read frame
        ret, img = video.read()

        if ret is True:
            # Display the resulting frame
            cv2.imshow('Frame', img)

            # write frame
            result.write(img)

            # break condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # release video
    video.release()
    result.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    fire.Fire(test_live_video)
