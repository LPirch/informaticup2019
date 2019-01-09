#!/usr/bin/python
from project_conf import API_KEY_LOCATION, DATA_ROOT, LOG_DIR, REMOTE_URL, GTSRB_PKL_PATH
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


def load_api_key():
	with open(API_KEY_LOCATION, 'r') as f:
		key = f.read().strip()
	return key


def test_connection(api_key):
	""" Test the connection for a given API key. 
		Returns 1 on success, 0 for an invalid API key and -1 for any other error (e.g. connectivity problems).
	"""
	try:
		r = requests.post(REMOTE_URL, data={'key': api_key}, files={'image': None})
		if int(r.status_code) == 401:
			return 0
		if int(r.status_code) == 400:
			return 1
	except:
		pass
	
	return -1

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
	api_key = load_api_key()
	assert api_key

	preds = {}
	preds_pickle = GTSRB_PKL_PATH
	if os.path.exists(preds_pickle):
		with open(preds_pickle, 'rb') as pkl:
			preds = pickle.load(pkl)
	if img_name in preds:
		print("skipping ", img_name)
	else:
		print("fetching ", img_name)
		r = requests.post(REMOTE_URL, data={'key': api_key}, files={'image': img})
		
		if r.status_code != 200:
			log_http_error(r.status_code, r.text)

		json_data = json.loads(r.text)

		preds[img_name] = json_data
		with open(preds_pickle, 'wb') as pkl:
			pickle.dump(preds, pkl)
		
		if delay:
			time.sleep(delay)

def fetch_batch_prediction(imgs, id_map, n_classes, one_hot=False, delay=None, remote_pred_precision=3, cache=None):
	return np.array([fetch_single_prediction(img, id_map, n_classes, one_hot=one_hot, delay=delay, remote_pred_precision=remote_pred_precision, cache=cache, key=i) for i, img in enumerate(imgs)])


def fetch_single_prediction(img, id_map, n_classes, one_hot=False, delay=None, remote_pred_precision=3, cache=None, key=None):
	api_key = load_api_key()
	assert api_key

	if cache and key in cache:
		json_data = cache[key]
	else:
		#  print("cache miss, crawling img", key)
		if not os.path.exists(DATA_DIR):
			os.makedirs(DATA_DIR)

		img = np.rint(img * 255).astype('uint8')
		img = Image.fromarray(img, 'RGB')
		img.save('tmp.png')
		with open('tmp.png', 'rb') as img_bytes:
			r = requests.post(REMOTE_URL, data={'key': api_key}, files={'image': img_bytes})
		os.remove('tmp.png')

		if r.status_code != 200:
			log_http_error(r.status_code, r.text)

		# template: [ {'class_name_0': str, 'confidence': float}, [...], {'class_name_4': str, 'confidence': float} ]
		json_data = json.loads(r.text)

		if cache:
			cache[key] = json_data

		if delay:
			time.sleep(delay)

	prediction = np.zeros(n_classes)
	if one_hot:
		# use only highest confidence label
		prediction[id_map[json_data[0]['class']]] = 1
	else:
		for pred in json_data:
			prediction[id_map[pred['class']]] = pred['confidence']

	return prediction.round(remote_pred_precision)


def log_http_error(status, text):
	with open(LOG_DIR+"/http_"+str(status)+".log", "a") as log:
		log.write(text+"\n")
	
if __name__ == "__main__":
	remote_evaluation(DATA_DIR)
