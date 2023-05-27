import numpy as np
import cv2
import fire
import time


def main():
    # current time
    start_time = time.time()
    
    # Load the video
    video = cv2.VideoCapture(0)

    # Read the first frame
    ret, old_frame = video.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask for drawing optical flow tracks
    mask = np.zeros_like(old_frame)

    # Iterate over frames and pass each for prediction
    while ret:
        # Read the next frame
        ret, frame = video.read()
        if not ret:
            break

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # display flow after x seconds
        if time.time() - start_time > 3:

            # Calculate optical flow between the frames
            p0 = cv2.goodFeaturesToTrack(old_gray,
                                         maxCorners=100,
                                         qualityLevel=0.1,
                                         minDistance=7)

            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                   frame_gray,
                                                   p0,
                                                   None)

            # Select only the good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b = int(a), int(b)
                c, d = int(c), int(d)
                mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)

            # Overlay the optical flow tracks on the frame
            output = cv2.add(frame, mask)

            # Display the resulting frame
            cv2.imshow('Optical Flow', output)

        else:  # display original frame
            cv2.imshow('Optical Flow', frame)

        # Update the previous frame and points
        old_gray = frame_gray.copy()

        # break condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print("The video was successfully saved")


if __name__ == "__main__":
    fire.Fire(main)
