
import keras.backend as K
from cleverhans.utils_keras import KerasModelWrapper
from keras.models import load_model

import numpy as np

import tensorflow as tf
import logging
import pickle
import sys

import os
import os.path
from PIL import Image

from utils import Timer
from gtsrb import GTSRB

from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, LBFGS, SPSA, ProjectedGradientDescent, SaliencyMapMethod
from cleverhans.utils import set_log_level

# TODO: re-add physical attack
from robust_physical_perturbations.attack import Physical, softmax

from nn_robust_attacks.l2_attack_robust import CarliniL2Robust

FLAGS = tf.flags.FLAGS

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

	print("Loaded pickle file", flush=True)

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
				label_map[class_id] = key

	print("Create label map", flush=True)

	if FLAGS.generate_random:
		print("Using random noise")
		img = np.random.rand(FLAGS.img_size, FLAGS.img_size, 3)
	else:
		print("Loading image from", FLAGS.image)
		with Image.open(FLAGS.image) as img:
			img = dataset.preprocess(img)

	# setup session
	sess = tf.Session()
	K.set_session(sess)

	# setup tf input placeholder
	x = tf.placeholder(tf.float32, shape=(None, dataset.img_size, dataset.img_size, dataset.n_channels))

	# load model
	tf_model = load_model(FLAGS.model_folder + FLAGS.model + ".h5", compile=False)
	model = KerasModelWrapper(tf_model)

	n_classes = tf_model.output_shape[1]

	# symbolic model predictions
	logits = model.get_logits(x)

	print("Target: ", FLAGS.target, label_map[FLAGS.target])

	adv_inputs = np.array([img])
	adv_targets = np.expand_dims(np.eye(n_classes)[FLAGS.target], axis=0)

	# attack dict
	# TODO: maybe put this in a separate config file
	attacks = {
		'cwl2': CarliniWagnerL2,
		'fgsm': FastGradientMethod,
		'lbfgs': LBFGS,
		'spsa': SPSA,
		'pgd': ProjectedGradientDescent,
		'jsma': SaliencyMapMethod,
		'physical': Physical,
		'robust_cwl2': CarliniL2Robust
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
		},
		'physical': {
			'y_target': adv_targets,
			'mask_path': FLAGS.mask_image,
			'max_iterations': FLAGS.max_iterations
		},
		'robust_cwl2': {
			'y_target': adv_targets,
			'max_iterations': FLAGS.max_iterations,
			'binary_search_steps': FLAGS.binary_search_steps,
			'learning_rate': 0.01,
			'batch_size': 1,
			'initial_const': 10,
			'confidence': FLAGS.confidence,
			'clip_min': FLAGS.boxmin,
			'clip_max': FLAGS.boxmax,
			'num_labels': n_classes,
			'outdir': FLAGS.outdir,
		}
	}

	# setup the attack
	# TODO: port physical to cleverhans interface
	attack = attacks[FLAGS.attack](model, sess=sess)
	attack_kwargs = attack_params[FLAGS.attack]

	print("Starting attack", flush=True)
	print("Parameters: ", flush=True)

	for k, v in attack_kwargs.items():
		print(k,":", v)
	print("", flush=True)

	# attack images
	with Timer("Attack (n_images=" + str(len(adv_inputs)) + ")"):
		adv = attack.generate_np(adv_inputs, **attack_kwargs)

	print("Attack finished", flush=True)

	# prepare img data for writing to file
	inputs_img = np.rint(adv_inputs * 255).astype('uint8')
	adv_img = np.rint(adv * 255).astype('uint8')

	outdir = FLAGS.outdir

	if not os.path.exists(outdir):
		os.makedirs(outdir)

	for i in range(len(adv)):
		filepath = os.path.join(outdir, FLAGS.attack + "_")
		print(filepath)

		# Original image
		img = Image.fromarray(inputs_img[i], 'RGB')
		img.save(filepath + "original.png")

		orig_y = model_logits(sess, x, logits, adv_inputs[i:i+1])
		pred_input_i = np.argmax(orig_y, axis=-1)
		adv_y = model_logits(sess, x, logits, adv[i:i+1])
		pred_adv_i = np.argmax(adv_y, axis=-1)

		if pred_adv_i != FLAGS.target:
			print("No adv: ", label_map[pred_input_i], label_map[pred_adv_i])
			continue

		# Adversarial images
		adv_image_path = filepath + str(pred_adv_i) + "adv.png"
		img = Image.fromarray(adv_img[i], 'RGB')
		img.save(adv_image_path)

		if not os.path.exists(adv_image_path):
			print("Saving file failed... retrying")
			img = Image.fromarray(adv[i], 'RGB')
			img.save(adv_image_path)

		if not os.path.exists(adv_image_path):
			print("Saving file failed again")
			print("Saving to pickle:")
			print(adv_image_path)
			with open(adv_image_path + ".pickle", "wb") as f:
				pickle.dump(f)

		print(label_map[pred_input_i], "->", label_map[pred_adv_i])
		print("Classification (original/target):", pred_input_i, "/", pred_adv_i)

		orig_softmax_y = softmax(orig_y)
		adv_softmax_y = softmax(adv_y)

		print("Original image: ")
		print(label_map[pred_input_i], orig_softmax_y[pred_input_i], "\t", label_map[pred_adv_i], orig_softmax_y[pred_adv_i])
		print("Adversarial image: ")
		print(label_map[pred_input_i], adv_softmax_y[pred_input_i], "\t", label_map[pred_adv_i], adv_softmax_y[pred_adv_i])

		print("Total distortion:", np.sum((adv[i]-adv_inputs[i])**2)**.5)

		with open(adv_image_path + ".conf", "w") as f:
			f.write("python3 " + " ".join(sys.argv))


if __name__ == '__main__':
	print("Starting attack", flush=True)

	tf.flags.DEFINE_integer("target", 0, "Target label")
	tf.flags.DEFINE_string("image", "", "Path to attacked image")
	tf.flags.DEFINE_string("mask_image", "masks/mask0.png", "Mask for image")
	tf.flags.DEFINE_boolean("generate_random", False, "Use random (noisy) image as source")

	tf.flags.DEFINE_integer("img_size", 64, "Image size")

	tf.flags.DEFINE_integer("batch_size", 1, "")
	tf.flags.DEFINE_integer("binary_search_steps", 3, "")
	tf.flags.DEFINE_integer("max_iterations", 1000, "")
	tf.flags.DEFINE_integer("confidence", 20, "")
	tf.flags.DEFINE_float("boxmin", 0, "")
	tf.flags.DEFINE_float("boxmax", 1, "")

	tf.flags.DEFINE_integer("debug_target_range", 43, "")
	tf.flags.DEFINE_integer("random_seed", 42, "")

	tf.flags.DEFINE_string("attack", "cwl2", "Type of attack")
	tf.flags.DEFINE_string("outdir", "/tmp", "Directory for saving images")
	tf.flags.DEFINE_string("model_folder", "model/trained/", "From where to load the model")
	tf.flags.DEFINE_string("model", "last-cleverhans_testmodel", "Trained model")

	main()
