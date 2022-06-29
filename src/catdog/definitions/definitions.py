import os

# server
IS_SERVER = False

if IS_SERVER:
    ROOT_DIR = '/alloc/data/fury_fda-fraud-evasion-off/'
else:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../../'

DATA_PATH = ROOT_DIR + 'data/'
MOCK_DATA_PATH = ROOT_DIR + 'tests/mock_data/'
IMG_PATH = DATA_PATH + 'images/'
CONFIG_NAME = 'config.yaml'
CONFIG_PATH = ROOT_DIR + 'conf/'


