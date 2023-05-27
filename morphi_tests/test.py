import numpy as np
import cv2
import fire
from transformers import pipeline
from PIL import Image

# Allocate a pipeline for object detection
model = "facebook/detr-resnet-50"
# model = 'hustvl/yolos-tiny'
object_detector = pipeline("object-detection", model=model)


# draw detections on a batch of frames
def make_detections(frame, object_detector):
    # Convert the frames to PIL Images
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform object detection on the frame
    detections = object_detector(frame_pil)
    return frame, detections


def draw_detections(frame, detections):
    # Draw the detections on the frame
    for detection in detections:
        x = int(detection['box']['xmin'])
        y = int(detection['box']['ymin'])
        width = int(detection['box']['xmax'] - detection['box']['xmin'])
        height = int(detection['box']['ymax'] - detection['box']['ymin'])
        label = detection['label']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame


def update_detections(frame, detections, flow):
    # propagate previous detection boxes using optical flow
    for detection in detections:
        xmin = int(detection['box']['xmin'])
        xmax = int(detection['box']['xmax'])
        ymin = int(detection['box']['ymin'])
        ymax = int(detection['box']['ymax'])

        # calculate optical flow for the box region
        flow_box = flow[ymin:ymax, xmin:xmax, :]
        flow_mean = np.mean(flow_box, axis=(0, 1))
        flow_x_mean = flow_mean[0]
        flow_y_mean = flow_mean[1]
        # update the box coordinates based on optical flow
        x_new = xmin + int(flow_x_mean)
        y_new = ymin + int(flow_y_mean)
        x_max_new = xmax + int(flow_x_mean)
        y_max_new = ymax + int(flow_y_mean)
        # draw the propagated box on the frame
        cv2.rectangle(frame, (x_new, y_new), (x_max_new, y_max_new), (0, 255, 0), 2)
        label = detection['label']
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame


def propagate_detections(frame, prev_detections, prev_keypoints, good_keypoints):
    # propagate detection boxes using optical flow
    propagated_boxes = []
    for i, (prev_pt, next_pt) in enumerate(zip(prev_keypoints, good_keypoints)):
        prev_x, prev_y = prev_pt.ravel()
        next_x, next_y = next_pt.ravel()

        # compute the displacement between the current and previous keypoints
        dx = next_x - prev_x
        dy = next_y - prev_y

        # propagate the detection box based on the displacement 
        xmin, ymin, xmax, ymax = prev_detections[i]['box']
        propagated_x = int(xmin + dx)
        propagated_y = int(ymin + dy)
        propagated_x_max = int(xmax + dx)
        propagated_y_max = int(ymax + dy)
        propagated_boxes.append((propagated_x,
                                 propagated_y,
                                 propagated_x_max,
                                 propagated_y_max))

        # draw the propagated box on the frame
        cv2.rectangle(frame, (propagated_x, propagated_y), (propagated_x_max, propagated_y_max), (0, 255, 0), 2)
        label = prev_detections[i]['label']
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    return frame, propagated_boxes


def main(savepath,
         resize_factor=0.25,
         record=False,
         optical_flow_threshold=5):

    video = cv2.VideoCapture(0)
    if not video.isOpened():
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

    # initialize previous frame and previous keypoints
    prev_frame = None
    prev_detections = []
    prev_keypoints = []
    good_keypoints = []
    started = 0

    # read until end of video
    while True:
        ret, frame = video.read()

        if ret:
            # downscale the frame
            frame = cv2.resize(frame,
                               (0, 0),
                               fx=resize_factor,
                               fy=resize_factor)

            # convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # if this is the first frame, we need to detect objects
            if prev_frame is not None:
                # calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_frame,
                                                    gray,
                                                    None,
                                                    0.5,
                                                    3,
                                                    15,
                                                    3,
                                                    5,
                                                    1.2,
                                                    0)

                # compute magnitude of the optical flow vectors
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

                # threshold the magnitude to identify regions with significant motion
                motion_mask = (magnitude > optical_flow_threshold).astype(np.uint8)
                

                # apply object detection only on frames with significant motion
                if np.any(motion_mask) or started == 0:
                    print("Motion detected")

                    # make detections
                    frame, detections = make_detections(frame, object_detector)

                    # draw detections
                    frame = draw_detections(frame, detections)

                    # update previous detections
                    prev_detections = detections

                    # update started flag
                    started += 1
                    started = min(started, 2)
                    
                    if started > 1:
                        # propagate previous keypoints using optical flow
                        print("Propagating keypoints")
                        print('prev_frame type', type(prev_frame))
                        print('gray type', type(gray))
                        
                        print('prev_keypoints type', prev_frame.shape)
                        print('gray type', gray.shape)
                        
                        next_keypoints, status, _ = \
                            cv2.calcOpticalFlowPyrLK(prev_frame,
                                                     gray,
                                                     prev_keypoints,
                                                     None)

                        # filter out keypoints with low status (indicating
                        # poor tracking)
                        good_keypoints = next_keypoints[status == 1]
                        prev_keypoints = prev_keypoints[status == 1]
                        
                else:
                    # propagate previous detection points using optical flow
                    frame, propagated_detections = \
                        propagate_detections(frame,
                                             prev_detections,
                                             prev_keypoints,
                                             good_keypoints)

                    # update previous detection boxes
                    prev_detections = propagated_detections
                    # frame = update_detections(frame, prev_detections, flow)

            # update previous frame and keypoints
            prev_frame = gray
            prev_keypoints = [np.array([kp], dtype=np.float32)
                              for kp in good_keypoints]

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Write the frame into the file savepath - must be an '.avi'
            if record:
                result.write(frame)

            # break condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    video.release()
    result.release()
    print("The video was successfully saved")


if __name__ == "__main__":
    fire.Fire(main)
