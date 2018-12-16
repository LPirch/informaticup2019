
import keras.backend as K
from cleverhans.utils_keras import KerasModelWrapper
from keras.models import load_model

import numpy as np

import tensorflow as tf
import logging
import pickle
import sys

from os import makedirs
from os.path import exists
from PIL import Image

from utils import Timer
from gtsrb import GTSRB

from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, LBFGS, SPSA, ProjectedGradientDescent, SaliencyMapMethod
from cleverhans.utils import set_log_level

# TODO: re-add physical attack
#from robust_physical_perturbations.attack import Physical

FLAGS = tf.flags.FLAGS


class GTSRBModel:
	def __init__(self, model, img_size, n_classes, session=None):
		self.num_channels = 3
		self.image_size = img_size
		self.num_labels = n_classes

		model.pop()
		self.model = model
	
	def predict(self, data):
		return self.model(data)


def model_logits(sess, x, predictions, samples, feed=None):
	"""
	Helper function that computes the current class logits
	:param sess: TF session
	:param x: the input placeholder
	:param logits: the model's symbolic logits
	:param samples: numpy array with input samples (dims must match x)
	:param feed: An optional dictionary that is appended to the feeding
				dictionary before the session runs. Can be used to feed
				the learning phase of a Keras model for instance.
	:return: the logits for each possible class
	"""
	feed_dict = {x: samples}
	if feed is not None:
		feed_dict.update(feed)
	logits = sess.run(predictions, feed_dict)

	if samples.shape[0] == 1:
		return logits[0]
	else:
		return logits


def main():
	with open("data/gtsrb.pickle", "rb") as f:
		gtsrb = pickle.load(f)
	dataset = GTSRB('data', FLAGS.random_seed)
	set_log_level(logging.DEBUG)
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
				label_map[class_id] = key.encode('latin-1')

	if FLAGS.generate_random:
		img = np.random.rand(FLAGS.img_size, FLAGS.img_size, 3)
	else:
		with Image.open(FLAGS.image) as img:
			img = dataset.preprocess(img)

	print("Target: ", FLAGS.target, label_map[FLAGS.target])

	adv_inputs = np.array([img])
	adv_targets = np.expand_dims(np.eye(FLAGS.n_classes)[FLAGS.target], axis=0)

	# setup session
	sess = tf.Session()
	K.set_session(sess)

	# setup tf input placeholder
	x = tf.placeholder(tf.float32, shape=(None, dataset.img_size, dataset.img_size, dataset.n_channels))

	# load model
	model = load_model("model/trained/" + FLAGS.model + ".h5", compile=False)
	model = KerasModelWrapper(model)

	# symbolic model predictions
	logits = model.get_logits(x)

	# attack dict
	# TODO: maybe put this in a separate config file
	attacks = {
		'cwl2': CarliniWagnerL2,
		'fgsm': FastGradientMethod,
		'lbfgs': LBFGS,
		'spsa': SPSA,
		'pgd': ProjectedGradientDescent,
		'jsma': SaliencyMapMethod
#		'physical': Physical
#		attack = Physical(sess, model, FLAGS.mask_image, max_iterations=FLAGS.max_iterations)
	}

	attack_params = {
		'cwl2': {
			'y_target': adv_targets,
			'max_iterations': FLAGS.max_iterations,
			'binary_search_steps': FLAGS.binary_search_steps,
			'learning_rate': 0.01,
			'batch_size': 1,
			'initial_const': 10,
			'confidence': FLAGS.confidence,
			'clip_min': FLAGS.boxmin,
			'clip_max': FLAGS.boxmax
		},
		'fgsm': {
			'y_target': adv_targets,
			'eps': 0.3,
			'ord': np.inf,
			'clip_min': FLAGS.boxmin,
			'clip_max': FLAGS.boxmax
		},
		'lbfgs': {
			'y_target': adv_targets,
			'max_iterations': FLAGS.max_iterations,
			'binary_search_steps': FLAGS.binary_search_steps,
			'batch_size': 1,
			'initial_const': 1e-2,
			'clip_min': FLAGS.boxmin,
			'clip_max': FLAGS.boxmax
		},
		'spsa': {},
		'pgd': {},
		'jsma': {
			'y_target': adv_targets,
			'theta': 1,
			'gamma': 0.1,
			'clip_min': FLAGS.boxmin,
			'clip_max': FLAGS.boxmax
		}
	}

	# setup the attack
	attack = attacks[FLAGS.attack](model, sess=sess)
	attack_kwargs = attack_params[FLAGS.attack]

	# attack images
	with Timer("Attack (n_images=" + str(len(adv_inputs)) + ")"):
		adv = attack.generate_np(adv_inputs, **attack_kwargs)

	# prepare img data for writing to file
	inputs_img = np.rint(adv_inputs * 255).astype('uint8')
	adv_img = np.rint(adv * 255).astype('uint8')

	outdir = "tmp/" + str(FLAGS.confidence) + "/"

	if not exists(outdir):
		makedirs(outdir)

	for i in range(len(adv)):
		filepath = outdir + FLAGS.attack + "_" + FLAGS.image[:-3] + "_"
		print(filepath)

		# Original image
		img = Image.fromarray(inputs_img[i], 'RGB')
		img.save(outdir + FLAGS.attack + "_" + FLAGS.image + "original.png")

		orig_y = model_logits(sess, x, logits, adv_inputs[i:i+1])
		pred_input_i = np.argmax(orig_y, axis=-1)
		adv_y = model_logits(sess, x, logits, adv[i:i+1])
		pred_adv_i = np.argmax(adv_y, axis=-1)

		if pred_adv_i != FLAGS.target:
			print("No adv: ", label_map[pred_input_i], label_map[pred_adv_i])
			continue

		# Adversarial images
		img = Image.fromarray(adv_img[i], 'RGB')
		img.save(filepath + str(pred_adv_i) + "adv.png")

		if not exists(filepath + str(pred_adv_i) + "adv.png"):
			print("Saving file failed... retrying")
			img = Image.fromarray(adv[i], 'RGB')
			img.save(filepath + str(pred_adv_i) + "adv.png")

		if not exists(filepath + str(pred_adv_i) + "adv.png"):
			print("Saving file failed again")
			print("Saving to pickle:")
			print(filepath + str(pred_adv_i) + "adv.png")
			with open(filepath + str(pred_adv_i) + "adv.pickle", "wb") as f:
				pickle.dump(f)

		print(label_map[pred_input_i], "->", label_map[pred_adv_i])
		print("Classification (original/target):", pred_input_i, "/", pred_adv_i)
		print("confidences: ", orig_y[pred_input_i], "/", orig_y[pred_adv_i], ",",
								adv_y[pred_input_i], "/", adv_y[pred_adv_i])

		print("Total distortion:", np.sum((adv[i]-adv_inputs[i])**2)**.5)

		with open(filepath + str(pred_adv_i) + ".conf", "w") as f:
			f.write("python3 " + " ".join(sys.argv))


if __name__ == '__main__':
	tf.flags.DEFINE_string("model", "last-cleverhans_testmodel", "Trained model")

	tf.flags.DEFINE_integer("target", 0, "Target label")
	tf.flags.DEFINE_string("image", "", "Path to attacked image")
	tf.flags.DEFINE_string("mask_image", "masks/mask0.png", "Mask for image")
	tf.flags.DEFINE_boolean("generate_random", False, "Use random (noisy) image as source")

	tf.flags.DEFINE_integer("img_size", 64, "Image size")
	tf.flags.DEFINE_integer("n_classes", 43, "Amount of classes")

	tf.flags.DEFINE_integer("batch_size", 1, "")
	tf.flags.DEFINE_integer("binary_search_steps", 3, "")
	tf.flags.DEFINE_integer("max_iterations", 10000, "")
	tf.flags.DEFINE_integer("confidence", 20, "")
	tf.flags.DEFINE_float("boxmin", 0, "")
	tf.flags.DEFINE_float("boxmax", 1, "")

	tf.flags.DEFINE_integer("debug_target_range", 43, "")
	tf.flags.DEFINE_integer("random_seed", 42, "")

	tf.flags.DEFINE_string("attack", "cwl2", "Type of attack")

	main()
