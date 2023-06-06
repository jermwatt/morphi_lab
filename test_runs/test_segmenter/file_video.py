import cv2
from morphi_lab.segmenter import Segmenter
from tqdm import tqdm
import fire


def segment_video(input_path, output_path):
    # read in video
    cap = cv2.VideoCapture(input_path)

    # get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # setup tqdm
    pbar = tqdm(total=total_frames)

    # get frame width and height from img
    success, img = cap.read()

    # get frame width and height from img
    frame_width = img.shape[1]
    frame_height = img.shape[0]

    # define codec and create VideoWriter object
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             30, size)

    # loop through video
    while success is not None:
        try:
            # segment frame
            segmenter = Segmenter(conf=0.15)
            segmenter.read_img(img)
            segmenter.segment()
            segmenter.project_segmentations()

            # write frame
            result.write(segmenter.img)
        except Exception as e:
            print(e, flush=True)
            break

        # read in next frame
        ret, img = cap.read()

        # update tqdm
        pbar.update(1)

    # release video
    cap.release()
    result.release()
    cv2.destroyAllWindows()

    # close tqdm
    pbar.close()


if __name__ == '__main__':
    fire.Fire(segment_video)
