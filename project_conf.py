import os

DATA_ROOT = 'data'
MODEL_SAVE_PATH = os.path.join(DATA_ROOT, 'models')
CACHE_DIR = os.path.join(DATA_ROOT, '.cache')
PROCESS_DIR = os.path.join(DATA_ROOT, '.process')
LOG_DIR = os.path.join(DATA_ROOT, 'logs')

API_KEY_LOCATION = 'api_key'
IMG_TMP_DIR = os.path.join(DATA_ROOT, "static_img")

TRAINING_PREFIX = "train"
ATTACK_PREFIX = "attack"

RANDOM_SEED = 42
REMOTE_URL = "https://phinau.de/trasi"