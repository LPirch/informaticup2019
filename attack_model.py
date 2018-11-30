import keras
import keras.backend as K
from keras.models import load_model, Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import zipfile
import time
import pickle
import random
from PIL import Image
import numpy as np
from io import BytesIO
from skimage import io
from train_model import preprocess_img
import tensorflow as tf
from fgsm import FGSM
from os.path import exists
from os import makedirs

from nn_robust_attacks.l2_attack import CarliniL2


class GTSRBModel:
	def __init__(self, model, n_labels, session=None):
		self.num_channels = 3
		self.image_size = 64
		self.num_labels = n_labels

		print(model.summary())
		self.model = model;
	
	def predict(self, data):
		return self.model(data)

class GTSRB:
	def __init__(self, imgs, labels):
		self.test_data = imgs
		self.test_labels = labels

		with open("data/gtsrb.pickle", "rb") as f:
			gtsrb = pickle.load(f)

		class_map = {}
		for filename, classification in sorted(gtsrb.items()):
			for c in classification:
				key = c["class"]
				if key not in class_map:
					class_map[key] = len(class_map)

		def get_class(img_path):
			filepath = "./data/" + img_path[:-3] + "png"

			output = np.zeros(43)

			for top_classification in gtsrb[filepath]:
				conf = top_classification["confidence"]
				label = top_classification["class"]

				one_hot = np.eye(43)[class_map[label]]

				output += one_hot * conf

			return output
		imgs, labels = [], []

		return

		# Extract training data
		with zipfile.ZipFile("data/GTSRB_Final_Training_Images.zip") as z:
			files = [name for name in z.namelist() if name.endswith(".png")]
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
		n_samples = len(imgs)
		self.train_data = X[:(n_samples*8)//10]
		self.train_labels = Y[:(n_samples*8)//10]
		self.validation_data = X[(n_samples*8)//10:]
		self.validation_data = Y[(n_samples*8)//10:]


def generate_data(data, samples, targeted=True, start=0, inception=False):
	"""
	Generate the input data to the attack algorithm.

	data: the images to attack
	samples: number of samples to use
	targeted: if true, construct targeted attacks, otherwise untargeted attacks
	start: offset into data to use
	inception: if targeted and inception, randomly sample 100 targets intead of 1000
	"""
	inputs = []
	targets = []
	for i in range(samples):
		if targeted:
			print(">>>>",data.test_labels)
			seq = range(43)

			for j in seq:
				if (j == data.test_labels[start+i]):
					continue
				inputs.append(data.test_data[start+i])
				targets.append(np.eye(43)[j])
		else:
			inputs.append(data.test_data[start+i])
			targets.append(data.test_labels[start+i])

	inputs = np.array(inputs)
	targets = np.array(targets)

	return inputs, targets

def main():
	with open("data/gtsrb.pickle", "rb") as f:
		gtsrb = pickle.load(f)

	class_map = {}
	label_map = {}
	for filename, classification in sorted(gtsrb.items()):
		for c in classification:
			key = c["class"]
			if key not in class_map:
				class_id = len(class_map)
				class_map[key] = class_id
				label_map[class_id] = key

	def get_class(img_path):
		filepath = "./data/" + img_path[:-3] + "png"
		label = gtsrb[filepath][0]["class"]

		return class_map[label]

	imgs = []
	labels = []
	with zipfile.ZipFile("data/GTSRB_Final_Test_Images.zip") as z:
		files = [name for name in z.namelist() if name.endswith(".png")]
		random.shuffle(files)
		files = files[:10]
		for i, name in enumerate(files):
			print(i)
			with z.open(name) as f:
				img = io.imread(BytesIO(f.read()))
				img = preprocess_img(img)

				imgs.append(img)
				labels.append(get_class(name))

	imgs = np.array(imgs)
	labels = np.array(labels)

	data = GTSRB(imgs, labels)

	sess = tf.Session()
	K.set_session(sess)

	model = load_model("model/trained/last-lukas_model.h5", compile=False)
	model = GTSRBModel(model, 43, session=sess)
	#attack = CarliniL2(sess, model, batch_size=10, max_iterations=1000, confidence=0)
	attack = FGSM(sess, model)

	inputs, targets = generate_data(data, samples=1, targeted=True, start=0, inception=False)

	timestart = time.time()
	adv = attack.attack(inputs, targets)
	timeend = time.time()

	print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

	inputs = np.rint(inputs * 255).astype('uint8')
	adv = np.rint(adv * 255).astype('uint8')

	if not exists("tmp/out"):
		makedirs("tmp/out")
	for i in range(len(adv)):
		print("Valid:")
		#plt.imshow(inputs[i])
		#plt.show()
		img = Image.fromarray(inputs[i], 'RGB')
		img.save("tmp/out/original.png")
		print("Adversarial:")
		#plt.imshow(adv[i])
		#plt.show()
		img = Image.fromarray(adv[i], 'RGB')
		img.save("tmp/out/adv"+str(i)+".png")
		
		pred_input = model.model.predict(inputs[i:i+1])[0]
		pred_input_i = np.argmax(pred_input)
		pred_adv = model.model.predict(adv[i:i+1])[0]
		pred_adv_i = np.argmax(pred_adv)
		print(i, ">>", label_map[pred_input_i])
		print("Classification (original/target):", pred_input_i, "/", pred_adv_i)
		print("confidences: ", pred_input[pred_input_i], "/", pred_input[pred_adv_i], ",", 
					pred_adv[pred_input_i], "/", pred_adv[pred_adv_i])

		print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)

if __name__ == '__main__':
	main()