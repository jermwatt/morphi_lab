import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
dir_path = os.path.dirname(os.path.realpath(__file__))

estimator = pipeline("depth-estimation", model='Intel/dpt-large', verbose=False)


class DepthEstimator:
    def __init__(self, segmenter):
        self.estimator = estimator
        self.segmenter = segmenter
        cv2.imwrite(self.segmenter.detection_window_path, self.segmenter.detection_window)
        
    def compute_depth(self):
        if self.segmenter.xmin is None:
            return self.segmenter.img

        # get depth map for detection window
        depth_map = self.estimator(self.segmenter.detection_window_path)
        self.depth_map = np.array(depth_map['depth'])
        
    # replace detection window with depth map
    def replace_detection_window_with_depth(self):   
        self.segmenter.img[self.segmenter.ymin:self.segmenter.ymax, self.segmenter.xmin:self.segmenter.xmax] = np.stack(( self.depth_map,) * 3, axis=-1)

    def replace_segmentation_with_depth(self):
        # get segmentation for detection window
        seq_forward = self.segmenter.seg[self.segmenter.ymin:self.segmenter.ymax, self.segmenter.xmin:self.segmenter.xmax]

        # create segmentation mask for depth map
        seg_depth = seq_forward * self.depth_map
        seg_depth = np.stack((seg_depth,) * 3, axis=-1)

        # create unmasked segmentation mask for detection window
        seq_zero_inds = np.where(seq_forward == 0)
        seq_nonzero_inds = np.where(seq_forward != 0)
        seq_backward = np.zeros_like(seq_forward)
        seq_backward[seq_zero_inds] = 1
        seq_backward[seq_nonzero_inds] = 0
        seq_backward = np.stack((seq_backward,) * 3, axis=-1)
        unmasked_detection = self.segmenter.detection_window * seq_backward

        # make copy of img to avoid modifying original        
        self.segmenter.img[self.segmenter.ymin:self.segmenter.ymax, self.segmenter.xmin:self.segmenter.xmax] = seg_depth + unmasked_detection

    def show_result(self):
        # image_rgb = cv2.imshow('img', img)
        image_rgb = cv2.cvtColor(self.segmenter.img, cv2.COLOR_BGR2RGB)

        plt.imshow(image_rgb)
        plt.axis('off')  # optional: disable the axis
        plt.show()