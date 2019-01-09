from project_conf import DATA_ROOT, TENSORBOARD_LOGDIR, MODEL_SAVE_PATH

import os
import logging

import numpy as np
from keras.models import Model as KerasModel
import keras.backend as K

import tensorflow as tf
from cleverhans.utils_keras import KerasModelWrapper
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam, SGD
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.momentum import MomentumOptimizer

from cleverhans.loss import CrossEntropy
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval
from cleverhans.train import train

from gtsrb import GTSRB
from models.model_specs import cnn_model, dense_model
from preprocess.crawler import fetch_single_prediction

MODEL_TYPES = {
	'cnn_model': cnn_model,
	'dense_model': dense_model
}

def init_modeldir():
	modeldir = os.path.join(DATA_ROOT, 'models')
	if not os.path.exists(modeldir):
		os.makedirs(modeldir)
	return modeldir

def init_modelspecs():
	model_specs = os.path.join(DATA_ROOT, )
	if not os.path.exists(model_specs):
		os.makedirs(model_specs)

	for name, initializer in MODEL_TYPES.items():
		modelpath = os.path.join(DATA_ROOT, name+'.h5')
		if not os.path.exists(modelpath):
			model = initializer()
			model.compile()
			model.save(modelpath)

def train_rebuild(random_seed=42, modelname='testmodel',
					optimizer='sgd', learning_rate=1e-2, batch_size=64,
					epochs=1, loss='categorical_crossentropy', validation_split=0.2,
					dataset_name='gtsrb', load_augmented=False, enable_tensorboard=False,
					max_per_class=1000, keras_verbosity=1):
	modeldir = init_modeldir()
	modelpath = os.path.join(modeldir, modelname+'.h5')
	report = AccuracyReport()

	tf.set_random_seed(random_seed)
	set_log_level(logging.DEBUG)

	if tf.test.is_gpu_available():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
		sess_kwargs = dict(gpu_options=gpu_options)
	else:
		sess_kwargs = {}

	sess = tf.Session(config=tf.ConfigProto(**sess_kwargs))

	if dataset_name == 'gtsrb':
		dataset = GTSRB(random_seed)
	else:
		raise NotImplementedError()

	if os.path.exists(modelpath):
		os.remove(modelpath)

	model = cnn_model(dataset.img_size, dataset.n_classes)
	
	if optimizer == "adam":
		op = Adam(lr=learning_rate, decay=1e-6)
	elif optimizer == "sgd":
		# tf equivalent of keras SGD optimizer
		op = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

	def lr_schedule(epoch):
		return learning_rate * (0.1 ** int(epoch / 10))

	X, Y = dataset.get_training_data(load_augmented=load_augmented, hot_encoded=True, max_per_class=max_per_class)

	model.compile(loss=loss,
					optimizer=op,
					metrics=['accuracy'])

	callbacks = [
		LearningRateScheduler(lr_schedule),
		ModelCheckpoint(modelpath, save_best_only=True)
	]
	if enable_tensorboard:
		callbacks.append(TensorBoard(log_dir=TENSORBOARD_LOGDIR))
	model.fit(X, Y,
				batch_size=batch_size,
				epochs=epochs,
				validation_split=validation_split,
				verbose=keras_verbosity,
				callbacks=callbacks
			)

	model.save(os.path.join(modeldir, 'last-'+modelname+'.h5'))

	K.clear_session()


def custom_argmin(arr):
		""" Returns the index where the maximum of a list entry is minimal. 
			Ignores zero entries of there are non-zero entries. 
			(Used to find maximum confidence samples.)"""
		assert len(arr) > 0
		min_el = arr[0]
		min_i = 0

		for i, el in enumerate(arr):
			if max(el) < max(min_el):
				min_i = i
				min_el = el

		return min_i


def crawl_initial_set(dataset, pickle_path, n_per_class=5, max_tries=100, confidence_threshold=0.9):
	# retrieve n high confidence sample from each class
	imgs, labs = dataset.get_training_data(max_per_class=max_tries*n_per_class, shuffle=False)

	# remove samples from classes unknown to the remote model
	imgs, labs = zip(*filter(lambda xy: xy[1] not in unknown_labels, zip(imgs, labs)))

	# translate to remote label ids
	imgs, labs = zip(*map(lambda xy: (xy[0], remote_map[gtsrb_map[xy[1]]]), zip(imgs, labs)))

	# sort according to new labs
	imgs, labs = zip(*sorted(zip(imgs, labs), key=lambda xy: xy[1]))

	n_classes = len(set(labs))
	X = np.zeros((n_classes*n_per_class, dataset.img_size, dataset.img_size, dataset.n_channels))
	done = np.zeros(n_classes)
	Y = np.zeros((n_classes*n_per_class, n_classes))

	print("Crawling initial set. This might take a while.")
	for i, (im, lab) in enumerate(zip(imgs, labs)):
		if done[int(lab)]:
			continue

		pred = fetch_single_prediction(im, remote_map, n_classes, delay=1)
		slice_indices = range(lab*n_per_class, (lab+1)*n_per_class)
		i_min = lab*n_per_class + custom_argmin(Y[slice_indices])

		if np.max(pred) > np.max(Y[i_min]):
			X[i_min] = im
			Y[i_min] = pred

			if np.max(Y[lab*n_per_class + custom_argmin(Y[slice_indices])]) > confidence_threshold:
				done[lab] = 1
				print("finished class "+str(lab)+" with max confidences:")
				print(*[np.max(y) for y in Y[slice_indices]])
				print("remaining classes: ", n_classes - sum(done))
		
	print("="*80, 'FINISHED CRAWLING')
	print("minimum confidence: ", Y[custom_argmin(Y)])

	with open(pickle_path, 'wb') as pkl:
		pickle.dump([X, Y], pkl)


def get_initial_set(dataset, n_per_class=1, max_tries=100, hot_encoded=False, confidence_threshold=0.9):
	pickle_path = os.path.join(DATA_ROOT, 'jbda_initial_set.pkl')
	if not os.path.exists(pickle_path):
		crawl_initial_set(dataset, pickle_path, n_per_class=n_per_class, max_tries=max_tries, confidence_threshold=confidence_threshold)

	with open(pickle_path, 'rb') as pkl:
		X, Y = pickle.load(pkl)
		X = np.array(X)
		Y = np.array(Y)

	if hot_encoded:
		Y_hot = np.zeros(Y.shape)
		for i, y in enumerate(Y):
			Y_hot[np.argmax(y)] = 1
		Y = Y_hot

	return X, Y


def train_substitute(modelname='testmodel', lmbda=0.1, tau=2, n_jac_iteration=5, 
					n_per_class=1, enable_tensorboard=False, batch_size=64, descent_only=False):
	modeldir = init_modeldir()
	modelpath = os.path.join(modeldir, modelname+'.h5')
	set_log_level(logging.DEBUG)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	sess_kwargs = dict(gpu_options=gpu_options)

	gtsrb = GTSRB('data', random_seed=42)
	n_classes = int(len(remote_map)/2)

	# (0) init constants
	LMBDA = 0.1
	TAU = 2
	N_JAC_ITERATION = 4
	cache = {}

	# (1) select initial training set
	if descent_only:
		# we select only a few images per class but these should be classified
		# with high confidence by the remote model
		X, _ = gtsrb.get_training_data(max_per_class=n_per_class, n_per_class=n_per_class, confidence_threshold=0.95)
	else:
		X, _ = get_initial_set(gtsrb, hot_encoded=False, n_per_class=n_per_class, confidence_threshold=0)

	def lr_schedule(epoch):
		return 0.01 * (0.1 ** int(epoch / 25))

	for rho in range(N_JAC_ITERATION):
		print("=" * 80, "jacobian iteration: ", rho, "training set size: ", len(X))
		# for memory reasons, we must reinitialize the session in each iteration
		sess = tf.Session(config=tf.ConfigProto(**sess_kwargs))
		K.set_session(sess)

		wrap = KerasModelWrapper(cnn_model(gtsrb.img_size, n_classes))

		op = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
		wrap.model.compile(loss='categorical_crossentropy',
							optimizer=op,
							metrics=['accuracy'])

		# (2) specify architecture (already specified)
		x = tf.placeholder(tf.float32, shape=(None, *X.shape[1:]))
		initial_weights = wrap.model.get_weights()

		if descent_only:
			# own approach
			lmbda_coef = -1
		else:
			# lambda as described in the paper
			lmbda_coef = (-1) ** (rho % TAU)

		# (3) label data
		Y = np.zeros(shape=(len(X), n_classes))
		for i in range(len(X)):
			# take known labels from cache
			if i in cache:
				Y[i] = cache[i]
			else:
				pred = fetch_single_prediction(X[i], remote_map, n_classes, delay=1)
				cache[i] = pred
				Y[i] = pred

		# (4) fit model on current set
		wrap.model.set_weights(initial_weights)
		callbacks = [
			LearningRateScheduler(lr_schedule),
			ModelCheckpoint(modelpath, save_best_only=True)
		]
		if enable_tensorboard:
			callbacks.append(Tensorboard(log_dir=TENSORBOARD_LOGDIR))
		wrap.model.fit(X, Y,
						batch_size=64,
						epochs=(rho+1)*20,
						validation_split=0.2 if len(X) > 64*5 else 0,
						verbose=2,
						callbacks= callbacks
					)


		# (5) augment data
		logits = wrap.get_logits(x)
		jacobian = jacobian_graph(logits, x, n_classes)
		Y_sub = np.array([np.argmax(row) for row in Y])
		X = jacobian_augmentation(sess, x, X, Y_sub, jacobian,
						lmbda=(LMBDA * lmbda_coef))
		if os.path.exists(modelpath):
			os.remove(modelpath)
		wrap.model.save(modelpath)
		K.clear_session()
		del sess

		# free as much memory as we can
		gc.collect()