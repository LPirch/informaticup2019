#!/usr/bin/python
import os
import time
import shutil
import zipfile
import requests
from PIL import Image

# TODO: leave this hard coded?
GTSRB_IMG_SHAPE = (64, 64)
SRC_URL = 'http://benchmark.ini.rub.de/Dataset'
DATASET_FILES = [
	"GTSRB_Final_Training_Images.zip",
	"GTSRB_Final_Test_Images.zip"
]
DATA_DIR = "../data"

def fetch_url(url, save_path):
	filename = url.split('/')[-1]
	full_path = os.path.join(save_path, filename)

	#TODO: maybe try/catch connection error
	response = requests.get(url, stream=True)
	with open(full_path, 'wb') as f:
		for chunk in response.iter_content(chunk_size=1024):
			# filtering out keep-alive chunks
			if chunk:
				f.write(chunk)


def normalize_images(zip_path, target_shape, src_ext='.ppm', target_ext='.png'):
	tmp_dir = "tmp"
	if os.path.exists(tmp_dir):
		shutil.rmtree(tmp_dir)
	# create tmp dir
	os.mkdir(tmp_dir)

	# unzip data
	with zipfile.ZipFile(zip_path, 'r') as z:
		z.extractall(tmp_dir)
		zip_basedir = z.namelist()[0].split(os.sep)[0]


	os.remove(zip_path)

	# transform data
	with zipfile.ZipFile(zip_path, 'w') as z:
		for root, _, files in os.walk(os.path.join(tmp_dir, zip_basedir)):
			if root is tmp_dir:
				root = "."
			# remove first element in root (="tmp/")
			zip_root = os.path.join(*root.split(os.sep)[1:])

			for filename in files:
				full_path = os.path.join(root, filename)
				path_in_zip = os.path.join(zip_root, filename)
				# directly write non-image files back to zip
				if not src_ext in filename:
					z.write(full_path, arcname=path_in_zip)
					continue

				# transform image
				with Image.open(full_path) as img:
					img = img.resize(target_shape)
					
					# modify file ending to png
					mod_name = full_path[:-len(src_ext)] + target_ext
					mod_name_zip = path_in_zip[:-len(src_ext)] + target_ext
					img.save(mod_name)
					z.write(mod_name, arcname=mod_name_zip)
					os.remove(mod_name)
					os.remove(full_path)
	
	shutil.rmtree(tmp_dir)

				
def fetch_dataset(data_dir, src_url, target_files, img_shape):
	if not os.path.exists(data_dir):
		os.mkdir(data_dir)

	print("Fetching data sets:")
	start = time.time()
	for i, zip_file in enumerate(target_files):
		print("[%d]: %s" % (i, zip_file))
		fetch_url(src_url + '/' + zip_file, save_path=data_dir)
		normalize_images(os.path.join(data_dir, zip_file), target_shape=GTSRB_IMG_SHAPE)
	end = time.time()
	print("\t -- finished after %d seconds" % (end-start))


if __name__ == '__main__':
	fetch_dataset(DATA_DIR, SRC_URL, DATASET_FILES, GTSRB_IMG_SHAPE)
