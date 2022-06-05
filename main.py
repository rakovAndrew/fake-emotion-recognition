import configparser
import os
from os.path import join

import numpy as np
from feat import Fex
from feat.utils import read_feat

import video_utils

config = configparser.ConfigParser()
config.read('config.ini')

video_utils.save_video_frames_and_aus_activity('dataset/validation/video/fake/pleasure/Happy_Beautiful_Hispanic_Woman_Smiles_Trim.mp4')
