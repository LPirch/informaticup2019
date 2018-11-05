import keras
import keras.backend as K
from io import BytesIO
from keras.models import load_model, Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from skimage import color, exposure, transform, io

import tensorflow as tf

import pickle
import numpy as np
import random
import os
import zipfile

from PIL import Image
from utils import Timer

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
tf.flags.DEFINE_string("load_name", "my_model", "Model to load.")
tf.flags.DEFINE_string("save_name", "my_model", "Where to save the model.")
tf.flags.DEFINE_string("train_folder", "data", "Folder containing training data.")

tf.flags.DEFINE_boolean("stealing", True, "Steal remote model")
tf.flags.DEFINE_boolean("steal_onehot", True, "")

tf.flags.DEFINE_integer("img_size", 64, "Image size")
tf.flags.DEFINE_integer("n_classes", 43, "Amount of classes")

tf.flags.DEFINE_string("optimizer", "adam", "Optimizer")
tf.flags.DEFINE_integer("batch_size", 128, "Batch size")
tf.flags.DEFINE_integer("epochs", 1, "Epochs")
tf.flags.DEFINE_float("decay", 1e-6, "Decay")
tf.flags.DEFINE_float("momentum", 1e-6, "Momentum")
tf.flags.DEFINE_string("loss", "categorical_crossentropy", "Loss function")
tf.flags.DEFINE_float("validation_split", 0.2, "Validation split")

def lr_schedule(epoch):
	return FLAGS.learning_rate * (0.1 ** int(epoch / 10))


def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, (FLAGS.img_size, FLAGS.img_size))

    # roll color axis to axis 0
    #img = np.rollaxis(img, -1)

    return img


def main():
	model_path = "model/untrained/" + FLAGS.load_name + ".h5"

	if not os.path.exists(model_path):
		print(model_path, "does not exist")
		sys.exit(-1)

	if not os.path.exists("model/trained"):
		os.makedirs("model/trained")

	if os.path.exists("model/trained/" + FLAGS.save_name + ".h5"):
		print("model/trained/" + FLAGS.save_name + ".h5 already exists")
		sys.exit(-1)

	# Define get_class method, which is used to map
	# an image path to an [n_classes]-dimensional vector.
	# Each vector element represents the likeliness of
	# the respective class

	# if FLAGS.stealing is true, we use the output of
	# our remote model

	# if FLAGS.stealing is false, we construct the vector
	# using the image's filename

	if FLAGS.stealing:
		# Train the model on data which was classified by
		# a remote model (i.e. steal the remote model)

		# This file contains the data which was labeled by
		# the remote model
		with open("gtsrb.pickle", "rb") as f:
			gtsrb = pickle.load(f)

		# The labels are saved as strings which we need
		# to map to integer values for training
		class_map = {}
		for filename, classification in gtsrb.items():
			for c in classification:
				key = c["class"]
				if key not in class_map:
					class_map[key] = len(class_map)

		# Create a vector using the remote's model
		# classification
		def get_class(img_path):
			filepath = "./data/" + img_path[:-3] + "png"

			if FLAGS.steal_onehot:
				top_classification = gtsrb[filepath][0]
				conf = top_classification["confidence"]
				label = top_classification["class"]

				return np.eye(FLAGS.n_classes)[class_map[label]]
			else:
				output = np.zeros(FLAGS.n_classes)

				for top_classification in gtsrb[filepath]:
					conf = top_classification["confidence"]
					label = top_classification["class"]

					one_hot = np.eye(FLAGS.n_classes)[class_map[label]]

					output += one_hot * conf

				return output
	else:
		# Create a vector using the image path to infer
		# the label
		def get_class(img_path):
			return np.eye(FLAGS.n_classes)[int(img_path.split('/')[-2])]

	with tf.Session() as sess:
		K.set_session(sess)

		# All models are saved without the final softmax layer,
		# because it adds no information and hides the logits
		model_wo_sm = load_model(model_path)
		model = Sequential()
		model.add(model_wo_sm)
		model.add(Dense(FLAGS.n_classes, activation='softmax'))

		imgs, labels = [], []

		# Extract training data
		with zipfile.ZipFile("data/GTSRB_Final_Training_Images.zip") as z:
			files = [name for name in z.namelist() if name.endswith(".ppm")]
			random.shuffle(files)
			for i, name in enumerate(files):
				if i % (len(files) // 10) == 0:
					print(i)

				with z.open(name) as f:
					img = io.imread(BytesIO(f.read()))
					img = preprocess_img(img)

					imgs.append(img)
					labels.append(get_class(name))

		# Convert to numpy arrays
		X = np.array(imgs, dtype='float32')
		Y = np.array(labels)

		# Define optimizer for training
		if FLAGS.optimizer == "adam":
			op = Adam(lr=FLAGS.learning_rate, decay=FLAGS.decay)
		elif FLAGS.optimizer == "sgd":
			op = SGD(lr=FLAGS.learning_rate, decay=FLAGS.decay, momentum=FLAGS.momentum, nesterov=True)
		else:
			raise

		model.compile(loss=FLAGS.loss,
			optimizer=op,
			metrics=['accuracy'])

		model.fit(X, Y,
			batch_size=FLAGS.batch_size,
			epochs=FLAGS.epochs,
			validation_split=FLAGS.validation_split,
			callbacks=[
				LearningRateScheduler(lr_schedule),
				ModelCheckpoint("model/trained/" + FLAGS.load_name + ".h5", save_best_only=True),
				TensorBoard()
			]
		)


if __name__ == "__main__":
	main()
