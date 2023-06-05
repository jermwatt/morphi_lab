Setup for usage of yolo with lap for tracking

Step 1: create conda env with python 3.8

```bash 
conda create --name test python=3.8 
```

Step 2: activate conda env  

```bash  
conda activate test
```

Step 3: install requirements 

```bash 
pip install -r requirements.txt
```

Step 4: download a pretrained yolo8 model and place it in the `/models` directory.  For example, [yolo8 segmentation models](https://docs.ultralytics.com/tasks/segment/).


Step 5: start tracker

```python 
python track.py
```