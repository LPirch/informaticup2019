from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K

import tensorflow as tf

import os
import sys

K.set_image_data_format('channels_last')

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("img_size", 64, "Image size")
tf.flags.DEFINE_integer("n_classes", 43, "Amount of classes")

tf.flags.DEFINE_string("filepath", "my_model", "Where to save the model")
tf.flags.DEFINE_string("model_type", "default", "Type of the model")

def cnn_model_wo_softmax():
	img_size = FLAGS.img_size
	n_classes = FLAGS.n_classes

	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same',
		input_shape=(img_size, img_size, 3),
		activation='relu'))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, (3, 3), padding='same',
		activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(128, (3, 3), padding='same',
		activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))

	return model


def create_and_save(creator, filepath):
	model = creator()
	model.save(filepath)


function_mapping = {
	"cnn": cnn_model_wo_softmax,
	"default": cnn_model_wo_softmax
}


def main():
	filename = FLAGS.filepath
	filepath = "model/untrained/" + filename + ".h5"

	model_type = FLAGS.model_type

	if not os.path.exists("model"):
		os.makedirs("model")

	if not os.path.exists("model/untrained"):
		os.makedirs("model/untrained")

	if os.path.exists(filepath):
		print(filepath, "already exists")
		sys.exit(-1)

	if model_type in function_mapping:
		create_and_save(function_mapping[model_type], filepath)
	else:
		print("model not found - available models:")
		print(function_mapping.keys())
		sys.exit(-1)


if __name__ == "__main__":
	main()