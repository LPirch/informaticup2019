#!/usr/bin/python3

import os
import json
import time
import pickle
import requests
from PIL import Image

# the api key is obviously not put into the repo
API_KEY = None
IMG_SHAPE = (64, 64)
URL = "https://phinau.de/trasi"
ENCODING_TYPE = "multipart/form-data"
PREDS_PICKLE = "./data/gtsrb.pickle"
DATA_DIR = "./data/gtsrb"

def load_api_key(loc="./api_key"):
	with open(loc, 'r') as f:
		key = f.read().strip()
	return key

def apply_on_images(directory, fun, fun_args=[], fun_kwargs={}, filter_ext=None):
	for root, _, files in os.walk(directory):
		for file in files:
			# optionally filter files by substring
			if not filter_ext or file.endswith(filter_ext):
				file = os.path.join(root, file)
				fun(file, *fun_args, **fun_kwargs)

def resize_image(filename, target_shape):
	with Image.open(filename) as img:
		img = img.resize(target_shape)
		img.save(filename)

def convert_img(filename, target_ext=".png"):
	with Image.open(filename) as img:
		core_name = ".".join(filename.split(".")[:-1])
		img.save(core_name + target_ext)
		os.remove(filename)

def check_shape(filename):
	with Image.open(filename) as img:
		assert IMG_SHAPE == img.size

def fetch_oracle_response(filename, delay=None):
	preds = {}
	if os.path.exists(PREDS_PICKLE):
		with open(PREDS_PICKLE, 'rb') as pkl:
			preds = pickle.load(pkl)
	if filename in preds:
		print("skipping "+filename)
	else:
		print("fetching "+filename)
		with open(filename, 'rb') as img:
			r = requests.post(URL, data={'key': API_KEY}, files={'image': img})

		if r.status_code != 200:
			log_http_error(r.status_code, r.text)

		json_data = json.loads(r.text)

		preds[filename] = json_data
		with open(PREDS_PICKLE, 'wb') as pkl:
			pickle.dump(preds, pkl)
		
		if delay:
			time.sleep(delay)


def log_http_error(status, text):
	with open("./logs/http_"+str(status)+".log", "a") as log:
		log.write(text+"\n")
	
if __name__ == "__main__":
	API_KEY = load_api_key()
	
	# convert ppm files to png and resize them to requested shape
	apply_on_images(DATA_DIR, convert_img, fun_kwargs={"target_ext": ".png"}, filter_ext=".ppm")
	apply_on_images(DATA_DIR, resize_image, fun_args=[IMG_SHAPE], filter_ext=".png")
	apply_on_images(DATA_DIR, check_shape, filter_ext=".png")
	apply_on_images(DATA_DIR, fetch_oracle_response, fun_kwargs={"delay": 1}, filter_ext=".png")
