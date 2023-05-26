from pytube import YouTube
import cv2
from PIL import Image
from tqdm import tqdm


def download_youtube_video(video_url='https://www.youtube.com/shorts/VwvKpsldxFA', output_path=''):
    # Create a YouTube object
    yt = YouTube(video_url)

    # Get the highest resolution video stream
    stream = yt.streams.get_highest_resolution()

    # Download the video
    stream.download(output_path=output_path)

    print("Video downloaded successfully!")


def draw_detections(object_detector, frames):
    # Convert the frames to PIL Images
    frame_pils = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

    # Perform object detection on the frames
    all_detections = object_detector(frame_pils)

    for i, detections in enumerate(all_detections):
        frame = frames[i]
        for detection in detections:
            x = int(detection['box']['xmin'])
            y = int(detection['box']['ymin'])
            width = int(detection['box']['xmax'] - detection['box']['xmin'])
            height = int(detection['box']['ymax'] - detection['box']['ymin'])
            label = detection['label']
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frames


def detect_objects_in_video(object_detector, input_path, output_path, batch_size=1, frame_resize=1):
    # Load the video
    video = cv2.VideoCapture(input_path)

    # Get the video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, unit='frame')
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)

    # Adjust frame_width and frame_height if frame_resize is provided
    if frame_resize != 1:
        frame_width = int(frame_width * frame_resize)
        frame_height = int(frame_height * frame_resize)
        size = (frame_width, frame_height)

    # Create a VideoWriter object to save the output video
    result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

    # Read the video's first frame
    ret, frame = video.read()

    # Resize the frame if frame_resize is provided
    if frame_resize != 1:
        frame = cv2.resize(frame, (frame_width, frame_height))

    # Initialize a list to store frames for batch processing
    batch_frames = []

    # Iterate over frames and pass each for prediction
    while ret:
        # Add the frame to the batch_frames list
        batch_frames.append(frame)

        # If the batch size is reached or it's the last frame, process the batch
        if len(batch_frames) == batch_size or not ret:
            # Perform object detection on the batch of frames
            batch_frames = draw_detections(object_detector, batch_frames)

            # Write the processed frames to the output video
            for processed_frame in batch_frames:
                result.write(processed_frame)
                cv2.imshow('Frame', processed_frame)
                progress_bar.update(1)

            # Reset the batch_frames list for the next batch
            batch_frames = []

        # Read the next frame
        ret, frame = video.read()

        # Resize the frame if frame_resize is provided
        if ret and frame_resize != 1:
            frame = cv2.resize(frame, (frame_width, frame_height))

    # Release resources
    video.release()
    result.release()

    print("The video was successfully saved")
    progress_bar.close()