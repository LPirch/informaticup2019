#!/usr/bin/python

import os
import json
import time
import pickle
import zipfile
import requests
from PIL import Image
from io import BytesIO
from skimage import io

# the api key is obviously not put into the repo
API_KEY = None
DATA_DIR = "../data"
LOG_DIR = "../logs"
URL = "https://phinau.de/trasi"
ENCODING_TYPE = "multipart/form-data"
PREDS_PICKLE = DATA_DIR +"/gtsrb.pickle"
IMG_SHAPE = (64, 64)


def load_api_key(loc="../api_key"):
	with open(loc, 'r') as f:
		key = f.read().strip()
	return key


def remote_evaluation(directory, target_ext=".png"):
	for root, _, files in os.walk(directory):
		files = [name for name in files if name.endswith(".zip")]
		for filename in files:
			with zipfile.ZipFile(os.path.join(root, filename)) as z:
				namelist = [name for name in z.namelist() if name.endswith(target_ext)]
				for img_name in namelist:
					with z.open(img_name) as img_file:
						fetch_oracle_response(img_name, img_file, delay=1)


def fetch_oracle_response(img_name, img, delay=None):
	preds = {}
	if os.path.exists(PREDS_PICKLE):
		with open(PREDS_PICKLE, 'rb') as pkl:
			preds = pickle.load(pkl)
	if img_name in preds:
		print("skipping ", img_name)
	else:
		print("fetching ", img_name)
		r = requests.post(URL, data={'key': API_KEY}, files={'image': img})
		
		if r.status_code != 200:
			log_http_error(r.status_code, r.text)

		json_data = json.loads(r.text)

		preds[img_name] = json_data
		with open(PREDS_PICKLE, 'wb') as pkl:
			pickle.dump(preds, pkl)
		
		if delay:
			time.sleep(delay)


def log_http_error(status, text):
	with open(LOG_DIR+"/http_"+str(status)+".log", "a") as log:
		log.write(text+"\n")
	
if __name__ == "__main__":
	API_KEY = load_api_key()
	remote_evaluation(DATA_DIR)
