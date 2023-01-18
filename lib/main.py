from tkinter.messagebox import NO, YES
import cv2, os
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
import numpy as np
import pickle
import importlib
from math import floor
from faceboxes_detector import *
# from ..faceboxes_detector import *
import time

import pyrealsense2 as rs
import cv2
import csv
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from networks import *
import data_utils
from functions import *

# sys.path.append("..\FaceBoxesV2")
# # sys.path.append("FaceBoxesV2")
# from faceboxes_detector import *

os.system("python ./demo_video_test_stable.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py camera")
