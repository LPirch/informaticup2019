import keras
import keras.backend as K
from keras.models import load_model, Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import zipfile
import time
import pickle
import random

from os import makedirs
from os.path import exists
from PIL import Image
from io import BytesIO
from skimage import io

from train_model import preprocess_img
from nn_robust_attacks.l2_attack import CarliniL2
from fgsm import FGSM
from utils import Timer

FLAGS = tf.flags.FLAGS

class GTSRBModel:
	def __init__(self, model, img_size, n_classes, session=None):
		self.num_channels = 3
		self.image_size = img_size
		self.num_labels = n_classes

		model.pop()
		self.model = model;
	
	def predict(self, data):
		return self.model(data)

class GTSRB:
	def __init__(self, imgs, labels):
		self.test_data = imgs
		self.test_labels = labels

def generate_data(data, samples, targeted=True, start=0, inception=False):
	"""
	Generate the input data to the attack algorithm.

	data: the images to attack
	samples: number of samples to use
	targeted: if true, construct targeted attacks, otherwise untargeted attacks
	start: offset into data to use
	inception: if targeted and inception, randomly sample 100 targets intead of 1000
	"""
	inputs = []
	targets = []
	for i in range(samples):
		data_label = np.argmax(data.test_labels[start+i])

		if targeted:
			seq = range(FLAGS.debug_target_range)

			for j in seq:
				if (j == data_label):
					continue
				inputs.append(data.test_data[start+i])
				targets.append(np.eye(FLAGS.n_classes)[j])
		else:
			inputs.append(data.test_data[start+i])
			targets.append(data_label)

	inputs = np.array(inputs)
	targets = np.array(targets)

	return inputs, targets

def main():
	with open("data/gtsrb.pickle", "rb") as f:
		gtsrb = pickle.load(f)

	# Create label map and class map
	class_map, label_map = {}, {}

	# Sorting must be applied to gtsrb, because the
	# mapping needs to be stable through restarts
	for filename, classification in sorted(gtsrb.items()):
		for c in classification:
			key = c["class"]
			if key not in class_map:
				class_id = len(class_map)
				class_map[key] = class_id
				label_map[class_id] = key

	# Define onehot_label method, which is used to map
	# an image path to an [n_classes]-dimensional vector.
	# Each vector element represents the likeliness of
	# the respective class

	def onehot_label(img_path):
		filepath = "./data/" + img_path[:-3] + FLAGS.pickle_extension

		top_classification = gtsrb[filepath][0]
		conf = top_classification["confidence"]
		label = top_classification["class"]

		return np.eye(FLAGS.n_classes)[class_map[label]]

	# load images and labels
	imgs, labels = [], []
	with zipfile.ZipFile("data/GTSRB_Final_Test_Images.zip") as z:
		files = [name for name in z.namelist() if name.endswith("." + FLAGS.zip_extension)]
		random.shuffle(files)
		files = files[:1]
		for i, name in enumerate(files):
			with z.open(name) as f:
				img = io.imread(BytesIO(f.read()))
				img = preprocess_img(img)

				imgs.append(img)
				labels.append(onehot_label(name))

	imgs = np.array(imgs)
	labels = np.array(labels)

	# setup session
	sess = tf.Session()
	K.set_session(sess)

	# load model
	model = load_model("model/trained/" + FLAGS.model + ".h5", compile=False)
	model = GTSRBModel(model, FLAGS.img_size, FLAGS.n_classes, session=sess)

	# setup the attack
	attack = CarliniL2(sess,
		model,
		batch_size=FLAGS.batch_size,
		binary_search_steps=FLAGS.binary_search_steps,
		max_iterations=FLAGS.max_iterations,
		confidence=FLAGS.confidence,
		boxmin=0,
		boxmax=1
	)
	#attack = FGSM(sess, model)

	# generate data for the attack
	data = GTSRB(imgs, labels)
	inputs, targets = generate_data(data, samples=1, targeted=True, start=0, inception=False)

	# attack images
	with Timer("Attack (n_images=" + str(len(inputs)) + ")"):
		adv = attack.attack(inputs, targets)

	inputs = np.rint(inputs * 255).astype('uint8')
	adv = np.rint(adv * 255).astype('uint8')

	if not exists("tmp/out"):
		makedirs("tmp/out")

	for i in range(len(adv)):
		# Original image
		img = Image.fromarray(inputs[i], 'RGB')
		img.save("tmp/out/original.png")

		# Predict original images
		pred_input = model.model.predict(inputs[i:i+1])[0]
		pred_input_i = np.argmax(pred_input)

		# Predict adversarial images
		pred_adv = model.model.predict(adv[i:i+1])[0]
		pred_adv_i = np.argmax(pred_adv)

		if (pred_input_i == pred_adv_i):
			print("No adv: ", label_map[pred_input_i], label_map[pred_adv_i])
			continue

		# Adversarial images
		img = Image.fromarray(adv[i], 'RGB')
		img.save("tmp/out/adv"+str(i)+".png")
		
		print(label_map[pred_input_i], "->", label_map[pred_adv_i])
		print("Classification (original/target):", pred_input_i, "/", pred_adv_i)
		print("confidences: ", pred_input[pred_input_i], "/", pred_input[pred_adv_i], ",", 
					pred_adv[pred_input_i], "/", pred_adv[pred_adv_i])

		print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)

if __name__ == '__main__':
	tf.flags.DEFINE_string("zip_extension", "ppm", "Extension of loaded images in zip")
	tf.flags.DEFINE_string("pickle_extension", "png", "Extension of loaded images in pickle")
	tf.flags.DEFINE_string("model", "last-lukas_model", "Trained model")

	tf.flags.DEFINE_integer("img_size", 64, "Image size")
	tf.flags.DEFINE_integer("n_classes", 43, "Amount of classes")

	tf.flags.DEFINE_integer("batch_size", 1, "")
	tf.flags.DEFINE_integer("binary_search_steps", 10, "")
	tf.flags.DEFINE_integer("max_iterations", 10000, "")
	tf.flags.DEFINE_integer("confidence", 20, "")

	tf.flags.DEFINE_integer("debug_target_range", 43, "")

	main()