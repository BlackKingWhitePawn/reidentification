from os.path import join
import pathlib

# debug
DEBUG = False

# enviroment
DATA_PATH = join(pathlib.Path().resolve(), 'data')
RESULTS_PATH = join(pathlib.Path().resolve(), 'results')

# constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# data statistics
MOT20_EXT_FIRST_AXIS_MEAN = 139
MOT20_EXT_SECOND_AXIS_MEAN = 62
MOT20_EXT_MEAN = None
MOT20_EXT_STD = None
