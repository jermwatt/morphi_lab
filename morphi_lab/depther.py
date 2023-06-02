import os
import cv2
import numpy as np
from transformers import pipeline
dir_path = os.path.dirname(os.path.realpath(__file__))

estimator = pipeline("depth-estimation", model='Intel/dpt-large', verbose=False)


class DepthEstimator:
    def __init__(self, segmenter):
        self.estimator = estimator
        self.segmenter = segmenter
        cv2.imwrite(self.segmenter.detection_window_path, self.segmenter.detection_window)

    def replace_segmentation_with_depth(self):
        if self.segmenter.xmin is None:
            return self.segmenter.img

        depth_map = self.estimator(self.segmenter.detection_window_path)
        depth_map = np.array(depth_map['depth'])

        seq_forward = self.segmenter.seg[self.segmenter.ymin:self.segmenter.ymax, self.segmenter.xmin:self.segmenter.xmax]

        seg_depth = seq_forward * depth_map
        seg_depth = np.stack((seg_depth,) * 3, axis=-1)

        seq_zero_inds = np.where(seq_forward == 0)
        seq_nonzero_inds = np.where(seq_forward != 0)
        seq_backward = np.zeros_like(seq_forward)
        seq_backward[seq_zero_inds] = 1
        seq_backward[seq_nonzero_inds] = 0
        seq_backward = np.stack((seq_backward,) * 3, axis=-1)

        unmasked_detection = self.segmenter.detection_window * seq_backward

        # make copy of img to avoid modifying original        
        self.segmenter.img[self.segmenter.ymin:self.segmenter.ymax, self.segmenter.xmin:self.segmenter.xmax] = seg_depth + unmasked_detection
