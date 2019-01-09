import os

CACHE_DIR = '.cache'
DATA_ROOT = 'data'
MODEL_SAVE_PATH = os.path.join(DATA_ROOT, 'models')
PROCESS_DIR = ".process"
IMG_TMP_DIR = os.path.join("static", "img")
RANDOM_SEED = 42
TRAINING_PREFIX = "train"
API_KEY_LOCATION = 'api_key'
LOG_DIR = "logs"

REMOTE_URL = "https://phinau.de/trasi"
GTSRB_PKL_PATH = os.path.join(DATA_ROOT, 'gtsrb.pkl')