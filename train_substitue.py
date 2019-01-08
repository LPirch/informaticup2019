import tensorflow as tf
from models.train import train_substitute

FLAGS = tf.flags.FLAGS

if __name__ == "__main__":
	tf.flags.DEFINE_string("modelname", "testmodel", "Name of the model.")
	tf.flags.DEFINE_boolean("enable_tensorboard", False, "Whether to enable tensorboard")
	tf.flags.DEFINE_float("lmbda", 0.1, "Step size to generate synthetic data along the remote gradient")
	tf.flags.DEFINE_integer("tau", 2, "Number of jacobian iteration after which to switch the sign of lambda")
	tf.flags.DEFINE_integer("n_jac_iteration", 5, "Number of jacobian iterations (how often to augment the data")
	tf.flags.DEFINE_integer("n_per_class", 1, "Number of samples to start augmentation with")
	tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training the substitute in the inner loop")
	tf.flags.DEFINE_boolean("descent_only", True, "Whether to start with high confidence samples and only generate lower confidence samples")

	train_dict = {
		"modelname":FLAGS.modelname ,
		"enable_tensorboard":FLAGS.enable_tensorboard ,
		"lmbda":FLAGS.lmbda ,
		"tau":FLAGS.tau ,
		"n_jac_iteration":FLAGS.n_jac_iteration ,
		"n_per_class":FLAGS.n_per_class ,
		"batch_size":FLAGS.batch_size,
		"descent_only": FLAGS.descent_only
	}

	train_substitute(**train_dict)	
