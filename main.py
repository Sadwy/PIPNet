from tkinter.messagebox import NO, YES
print('may I find you, cv2?@main.py')
import cv2, os
print(os.getcwd())
print('yes, you can.@main.py')
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '..')
import numpy as np
import pickle
import importlib
from math import floor
from FaceBoxesV2.faceboxes_detector import *
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

from lib.networks import *
import lib.data_utils
from lib.functions import *

os.system("python ./lib/demo_video_test_stable.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py camera")
