Setup for usage of yolo with lap for tracking

Step 1: create conda env with python 3.8

```bash 
conda create --name testies python=3.8 
```

Step 2: activate conda env  

```bash  
conda activate testies
```

Step 3: install requirements 

```bash 
pip install -r requirements.txt
```

Step 4: start tracker

```python 
python track.py
```