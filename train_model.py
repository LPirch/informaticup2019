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
from models import cnn_model

K.set_image_data_format('channels_last')

FLAGS = tf.flags.FLAGS


def lr_schedule(epoch):
	return FLAGS.learning_rate * (0.1 ** int(epoch / 10))


def main():
	report = AccuracyReport()

	# tf expects a fully qualified path (with leading './' and trailing '/' )
	#if FLAGS.train_dir:
	#   FLAGS.train_dir = os.path.join(".", FLAGS.train_dir, "")

	tf.set_random_seed(FLAGS.random_seed)
	set_log_level(logging.DEBUG)

	if tf.test.is_gpu_available():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
		sess_kwargs = dict(gpu_options=gpu_options)
	else:
		sess_kwargs = {}

	sess = tf.Session(config=tf.ConfigProto(**sess_kwargs))

	gtsrb = GTSRB('data', FLAGS.random_seed)
	"""
	x_train, x_test, y_train, y_test = gtsrb.get_training_data(load_augmented=FLAGS.load_augmented, hot_encoded=True,
																validation_split=FLAGS.validation_split)

	x = tf.placeholder(tf.float32, shape=(None, gtsrb.img_size, gtsrb.img_size, gtsrb.n_channels))
	y = tf.placeholder(tf.float32, shape=(None, gtsrb.n_classes))

	train_params = {
		'nb_epochs': FLAGS.epochs,
		'batch_size': FLAGS.batch_size,
		'learning_rate': FLAGS.learning_rate,
		'train_dir': FLAGS.train_dir,
		'filename': FLAGS.filename
	}

	rng = np.random.RandomState(42)
	"""

	if os.path.exists(FLAGS.filename):
		model = load_model(FLAGS.filename)
	else:
		model = cnn_model(gtsrb.img_size, gtsrb.n_classes)
	"""
	preds = model(x)

	def evaluate():
		# Evaluate the accuracy
		eval_params = {'batch_size': FLAGS.batch_size}
		acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
		report.clean_train_clean_eval = acc
		print('Test accuracy: %0.4f' % acc)
	
	# Define optimizer for training
	if FLAGS.optimizer == "adam":
		op = AdamOptimizer(learning_rate=FLAGS.learning_rate)
	elif FLAGS.optimizer == "sgd":
		# tf equivalent of keras SGD optimizer
		op = MomentumOptimizer(learning_rate=FLAGS.learning_rate,
								momentum=FLAGS.momentum,
								use_nesterov=True)
	else:
		raise NotImplementedError()

	loss = CrossEntropy(model, smoothing=FLAGS.label_smoothing)
	train(sess, loss, x_train, y_train, evaluate=evaluate, args=train_params,
										optimizer=op, rng=rng, use_ema=FLAGS.use_ema)

	model.keras_model.save(FLAGS.filename)
	sess.close()
	"""
	if FLAGS.optimizer == "adam":
		op = Adam(lr=FLAGS.learning_rate, decay=1e-6)
	elif FLAGS.optimizer == "sgd":
		# tf equivalent of keras SGD optimizer
		op = SGD(lr=FLAGS.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

	X, Y = gtsrb.get_training_data(load_augmented=FLAGS.load_augmented, hot_encoded=True)

	model.compile(loss=FLAGS.loss,
								optimizer=op,
								metrics=['accuracy'])

	model.fit(X, Y,
				batch_size=FLAGS.batch_size,
				epochs=FLAGS.epochs,
				validation_split=FLAGS.validation_split,
				callbacks=[
					LearningRateScheduler(lr_schedule),
					ModelCheckpoint("model/trained/" + FLAGS.filename + ".h5", save_best_only=True)
					#TensorBoard()
				]
			)

	model.save("model/trained/last-" + FLAGS.filename + ".h5")

	K.clear_session()

if __name__ == "__main__":
	tf.flags.DEFINE_string("filename", "cleverhans_testmodel", "Name of the model.")
	tf.flags.DEFINE_string("train_dir", "train_dir", "Folder where temporary training data is stored.")
	tf.flags.DEFINE_string("data_root", "data", "The path to the data set directory.")

	tf.flags.DEFINE_integer("random_seed", 42, "seed for random numbers")

	tf.flags.DEFINE_integer("epochs", 1, "Epochs")
	tf.flags.DEFINE_string("optimizer", "sgd", "Optimizer")
	tf.flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
	tf.flags.DEFINE_integer("batch_size", 64, "Batch size")
	tf.flags.DEFINE_float("momentum", 0.9, "Momentum")
	tf.flags.DEFINE_float("label_smoothing", 0.1, "The amout of label smoothing for cross entropy.")
	tf.flags.DEFINE_string("loss", "categorical_crossentropy", "Loss function")
	tf.flags.DEFINE_float("validation_split", 0.2, "Validation split")
	tf.flags.DEFINE_boolean("use_ema", False, "Whether to use exponential moving averages during training.")

	tf.flags.DEFINE_boolean("stealing", True, "Steal remote model")
	tf.flags.DEFINE_boolean("steal_onehot", False, "")
	tf.flags.DEFINE_boolean("load_model", True, "Whether to try to load an existing model or to train from scratch.")
	tf.flags.DEFINE_boolean("load_augmented", False, "")

	main()
