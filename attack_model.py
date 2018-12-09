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
import sys

from os import makedirs
from os.path import exists
from PIL import Image
from io import BytesIO
from skimage import io

from train_model import preprocess_img
from nn_robust_attacks.l2_attack import CarliniL2
from nn_robust_attacks.l0_attack import CarliniL0
from fgsm import FGSM
from utils import Timer

from robust_physical_perturbations.attack import Physical

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

	if FLAGS.generate_random:
		img = np.random.rand(FLAGS.img_size, FLAGS.img_size, 3)
	else:
		with Image.open(FLAGS.image) as img:
			img = np.asarray(img, dtype="uint8")
			img = img[:,:,:3]
			img = img.copy()

			# TODO: Use method from train_model.py
			assert img.shape == (FLAGS.img_size, FLAGS.img_size, 3)
			img = img / 255

	print("Target: ", FLAGS.target, label_map[FLAGS.target])

	# setup session
	sess = tf.Session()
	K.set_session(sess)

	# load model
	model = load_model("model/trained/" + FLAGS.model + ".h5", compile=False)
	model = GTSRBModel(model, FLAGS.img_size, FLAGS.n_classes, session=sess)

	# setup the attack
	if FLAGS.attack == "cwl2":
		attack = CarliniL2(sess,
			model,
			batch_size=FLAGS.batch_size,
			binary_search_steps=FLAGS.binary_search_steps,
			max_iterations=FLAGS.max_iterations,
			confidence=FLAGS.confidence,
			boxmin=FLAGS.boxmin,
			boxmax=FLAGS.boxmax
		)
	elif FLAGS.attack == "cwl0":
		attack = CarliniL0(sess, model, max_iterations=FLAGS.max_iterations)
	elif FLAGS.attack == "fgsm":
		attack = FGSM(sess, model, n_iterations=FLAGS.max_iterations)
	elif FLAGS.attack == "physical":
		attack = Physical(sess, model, FLAGS.mask_image, n_iterations=FLAGS.max_iterations)

		FLAGS.image = "generatedimage.png"
		FLAGS.confidence = "physical"
	else:
		raise RuntimeError("Unknown attack: " + FLAGS.attack)

	inputs = np.array([img])
	targets = np.array([np.eye(FLAGS.n_classes)[FLAGS.target]])

	# attack images
	with Timer("Attack (n_images=" + str(len(inputs)) + ")"):
		adv = attack.attack(inputs, targets)

	inputs = np.rint(inputs * 255).astype('uint8')
	adv = np.rint(adv * 255).astype('uint8')

	outdir = "tmp/" + str(FLAGS.confidence) + "/"

	if not exists(outdir):
		makedirs(outdir)

	for i in range(len(adv)):
		filepath = outdir + FLAGS.attack + "_" + FLAGS.image[:-3] + "_"
		print(filepath)

		# Original image
		img = Image.fromarray(inputs[i], 'RGB')
		img.save(outdir + FLAGS.attack + "_" + FLAGS.image + "original.png")

		# Predict original images
		pred_input = model.model.predict(inputs[i:i+1])[0]
		pred_input_i = np.argmax(pred_input)

		# Predict adversarial images
		pred_adv = model.model.predict(adv[i:i+1])[0]
		pred_adv_i = np.argmax(pred_adv)

		if FLAGS.attack == "physical":
			if pred_adv_i == pred_input_i:
				print("Source: {}, Adv: {}", label_map[pred_input_i], label_map[pred_adv_i])
				continue
		else:
			if (pred_adv_i != FLAGS.target):
				print("Source: {}, Target: {}, Adv: {}", label_map[pred_input_i], label_map[FLAGS.target], label_map[pred_adv_i])
				continue

		# Adversarial images
		img = Image.fromarray(adv[i], 'RGB')
		img.save(filepath + str(pred_adv_i) + "adv.png")

		print(label_map[pred_input_i], "->", label_map[pred_adv_i])
		print("Classification (original/target):", pred_input_i, "/", pred_adv_i)
		print("confidences: ", pred_input[pred_input_i], "/", pred_input[pred_adv_i], ",", 
					pred_adv[pred_input_i], "/", pred_adv[pred_adv_i])

		print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)

		with open(filepath + str(pred_adv_i) + ".conf", "w") as f:
			f.write("python3 " + " ".join(sys.argv))

if __name__ == '__main__':
	tf.flags.DEFINE_string("model", "last-lukas_model", "Trained model")

	tf.flags.DEFINE_integer("target", 0, "Target label")
	tf.flags.DEFINE_string("image", "", "Path to attacked image")
	tf.flags.DEFINE_string("mask_image", "mask_l1rectangles-more_64.png", "Mask for image")
	tf.flags.DEFINE_boolean("generate_random", False, "Use random (noisy) image as source")

	tf.flags.DEFINE_integer("img_size", 64, "Image size")
	tf.flags.DEFINE_integer("n_classes", 43, "Amount of classes")

	tf.flags.DEFINE_integer("batch_size", 1, "")
	tf.flags.DEFINE_integer("binary_search_steps", 10, "")
	tf.flags.DEFINE_integer("max_iterations", 200, "")
	tf.flags.DEFINE_integer("confidence", 20, "")
	tf.flags.DEFINE_float("boxmin", 0, "")
	tf.flags.DEFINE_float("boxmax", 1, "")

	tf.flags.DEFINE_integer("debug_target_range", 43, "")

	tf.flags.DEFINE_string("attack", "cwl2", "Type of attack")

	main()