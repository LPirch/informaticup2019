from project_conf import DATA_ROOT

import os
import logging

import numpy as np
from keras.models import Model as KerasModel
import keras.backend as K

import tensorflow as tf
from cleverhans.utils_keras import KerasModelWrapper
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
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

	model.fit(X, Y,
				batch_size=batch_size,
				epochs=epochs,
				validation_split=validation_split,
				verbose=keras_verbosity,
				callbacks=[
					LearningRateScheduler(lr_schedule),
					ModelCheckpoint(modelpath, save_best_only=True),
					#TensorBoard()
				]
			)

	model.save(os.path.join(modeldir, 'last-'+modelname+'.h5'))

	K.clear_session()

def train_substitute():
	pass