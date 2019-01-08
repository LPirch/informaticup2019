from project_conf import GTSRB_PKL_PATH

import keras
import keras.backend as K
from keras.models import load_model

import matplotlib.pyplot as plt

import tensorflow as tf

from io import BytesIO
from skimage import io
from PIL import Image

import sys
import zipfile
import csv
import numpy as np
import pickle
import random
import os.path

from gtsrb import GTSRB

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model", "gtsrb_model", "Trained model")
tf.flags.DEFINE_string("image", "", "Classified image")
tf.flags.DEFINE_integer("random_seed", 42, "")

with open(GTSRB_PKL_PATH, "rb") as f:
	gtsrb = pickle.load(f)

class_map = {}
label_map = {}
for filename, classification in sorted(gtsrb.items()):
	for c in classification:
		key = c["class"]
		if key not in class_map:
			class_id = len(class_map)
			class_map[key] = class_id
			label_map[class_id] = key

'''
def get_class(img_path):
	filepath = "./data/" + img_path[:-3] + "png"
	label = gtsrb[filepath][0]["class"]

	return class_map[label]

imgs = []
labels = []

with zipfile.ZipFile("data/GTSRB_Final_Training_Images.zip") as z:
	files = [name for name in z.namelist() if name.endswith(".ppm")]
	random.shuffle(files)
	files = files[:10]
	for i, name in enumerate(files):
		print(i)
		with z.open(name) as f:
			img = io.imread(BytesIO(f.read()))
			img = preprocess_img(img)

			imgs.append(img)
			labels.append(get_class(name))

imgs = np.array(imgs)
labels = np.array(labels)
'''

assert os.path.isfile(FLAGS.image)

dataset = GTSRB('data', FLAGS.random_seed)
with Image.open(FLAGS.image) as img:
	img = dataset.preprocess(img)

model = load_model("model/trained/" + FLAGS.model + ".h5")

# predict and evaluate
predictions = model.predict(np.array([img]))

print(np.max(predictions), np.argmax(predictions))

'''
cnt = 0
for img, (label, pred) in zip(imgs, zip(labels, predictions)):
	if label == np.argmax(pred):
		cnt += 1
	print(label, label_map[label], np.argmax(pred), label_map[np.argmax(pred)])
	plt.imshow(np.rollaxis(np.rollaxis(img, -1), -1))
	plt.show()

print(cnt)
print(len(labels))
print(cnt/len(labels))
'''