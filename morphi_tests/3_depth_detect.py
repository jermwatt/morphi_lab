import cv2
import fire
from transformers import pipeline
from utilities_video import draw_detection_single_frame


# Allocate a pipeline for object detection
model = "facebook/detr-resnet-50"
# model = 'hustvl/yolos-tiny'
object_detector = pipeline("object-detection", model=model)


def main(savepath,
         resize_factor=0.25,
         record=False):

    video = cv2.VideoCapture(0)
    if (video.isOpened() is False):
        print("Error reading video file")

    # get original frame width and height
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    # downscale the frame based on resize_factor
    frame_width = int(frame_width * resize_factor)
    frame_height = int(frame_height * resize_factor)

    # define codec and create VideoWriter object
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(savepath,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             30, size)

    # read until end of video
    while True:
        ret, frame = video.read()

        if ret is True:
            # downscale the frame
            frame = cv2.resize(frame, (0, 0),
                               fx=resize_factor,
                               fy=resize_factor)

            #### process frame ####
            # detect objects in the frame
            frame = draw_detection_single_frame(object_detector, frame)
            
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Write the frame into the file savepath - must be an '.avi'
            if record is True:
                result.write(frame)

            # break condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    video.release()
    result.release()
    cv2.destroyAllWindows()

    print("The video was successfully saved")


if __name__ == "__main__":
    fire.Fire(main)
