import tensorflow as tf
from models.train import train_rebuild
import functools

FLAGS = tf.flags.FLAGS

if __name__ == "__main__":
	tf.flags.DEFINE_string("modelname", "testmodel", "Name of the model.")
	tf.flags.DEFINE_string("modeltype", "cnn_model", "Type of the model.")
	tf.flags.DEFINE_string("dataset", "gtsrb", "The reference dataset")

	tf.flags.DEFINE_integer("random_seed", 42, "seed for random numbers")

	tf.flags.DEFINE_integer("epochs", 1, "Epochs")
	tf.flags.DEFINE_string("optimizer", "sgd", "Optimizer")
	tf.flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
	tf.flags.DEFINE_integer("batch_size", 64, "Batch size")
	tf.flags.DEFINE_float("momentum", 0.9, "Momentum")
	tf.flags.DEFINE_string("loss", "categorical_crossentropy", "Loss function")
	tf.flags.DEFINE_float("validation_split", 0.2, "Validation split")

	tf.flags.DEFINE_integer("max_per_class", 500, "Maximum number of images per class to load")
	tf.flags.DEFINE_boolean("load_augmented", False, "Whether to load the augmented dataset")
	tf.flags.DEFINE_integer("keras_verbosity", 1, "Set the verbosity of keras training output")

	train_dict = {
		'random_seed': FLAGS.random_seed,
		'modelname': FLAGS.modelname,
		'modeltype': FLAGS.modeltype,
		'optimizer': FLAGS.optimizer,
		'learning_rate': FLAGS.learning_rate,
		'batch_size': FLAGS.batch_size,
		'epochs': FLAGS.epochs,
		'loss': FLAGS.loss,
		'validation_split': FLAGS.validation_split,
		'dataset_name': FLAGS.dataset,
		'load_augmented': FLAGS.load_augmented,
		'max_per_class': FLAGS.max_per_class,
		'keras_verbosity': FLAGS.keras_verbosity
	}
	
	# set flush=True as default value for print
	print = functools.partial(print, flush=True)
	train_rebuild(**train_dict)	
