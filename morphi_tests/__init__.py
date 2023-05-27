import os
import sys

# path to this file's directory and parent directory
file_path = os.path.abspath(__file__)
base_directory = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

# add to path
sys.path.append(base_directory)
