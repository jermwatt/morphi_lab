# Morphi Lab

Morphi Lab is a project that explores the integration of image segmentation, depth estimation, and image generation techniques to enhance the generation of 2.5D images.

## Project Structure

The Morphi Lab repository has the following structure:

morphi_lab/: Contains models and classes for performing segmentation, depth estimation, and image generation.

- models/: Contains various detection and segmentation models.
- segmenter.py: Main segmentation class module.
- depther.py: Main depth estimation class module.
- test_runs/: Contains wrapper code for testing Morphi Lab functionality on images, saved videos, and live videos.


## Installation

This project utilizes [Python poetry](https://python-poetry.org/) for its installation process, following standard protocols.


## Functionality

Record video from laptop for testing 

```python
python test_runs/test_av/record_live_video.py --output_path=${PWD}/test_data/test_input/test_video_segmented.avi
```

Live cup segmentation 

```python 
python test_runs/test_segmenter/live_video.py  
```