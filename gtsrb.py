import os
import zipfile
import random
import warnings
import numpy as np
from skimage import io
from io import BytesIO
from skimage import transform
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class GTSRB():
	def __init__(self, data_root, random_seed):
		self.img_size = 64
		self.n_channels = 3
		self.n_classes = 43
		self.box_min = 0
		self.box_max = 1
		self.n_train_imgs = 39209
		self.n_test_imgs = 12630
		self.data_root = data_root
		self.train_zip = os.path.join(data_root, 'GTSRB_Final_Training_Images.zip')
		self.test_zip = os.path.join(data_root, 'GTSRB_Final_Test_Images.zip')
		self.img_extension = '.ppm'
		self.random_seed = random_seed

	def download_dataset(self):
		raise NotImplementedError()

	def get_class_distr(self, load_augmented=False):
		"""Return the class distribution in the training set."""

		class_distr = np.zeros(self.n_classes, dtype="uint32")

		# the data set should organize files of the same class in the same subdirectory
		with zipfile.ZipFile(self.train_zip) as z:
			files = filter(lambda x: x.endswith(self.img_extension), z.namelist())
			for f in files:
				class_id = self.get_class(f)
				class_distr[class_id] = class_distr[class_id] + 1

		if load_augmented:
			with zipfile.ZipFile(self.get_augmented_name()) as z:
				files = filter(lambda x: x.endswith(self.img_extension), z.namelist())
				for f in files:
					class_id = self.get_class(f)
					class_distr[class_id] = class_distr[class_id] + 1

		return class_distr

	def plot_class_distr(self, class_distr=None, output_file=None):
		"""Plots a histogramm of a given class distribution."""

		if class_distr is None:
			class_distr = self.get_class_distr()
		x = np.arange(len(class_distr))
		plt.bar(x, height=class_distr)
		plt.title('class distribution of data set')
		plt.xlabel('class id')
		if output_file:
			plt.savefig(output_file)
		else:
			plt.show()

	def get_training_data(self, shuffle=True, load_augmented=False, hot_encoded=False, max_per_class=1000,
							validation_split = None):
		"""Load the training data from the zip file and optionally load the augmented data.

		Keyword arguments:
		shuffle 		 -- whether to shuffle the image/label pairs (default: True)
		load_augmented 	 -- whether to include augmented data (default: False)
		hot_encoded 	 -- whether to return hot-encoded vectors as class labels
		max_per_class 	 -- the maximum number of training images per class (default: 1000)
		validation_split -- the percentage of validation samples
		"""
		n_per_class = np.zeros(self.n_classes, dtype="uint32")
		class_distr = self.get_class_distr(load_augmented=load_augmented)
		class_distr = np.clip(class_distr, a_min=0, a_max=max_per_class)

		# shuffle files to ensure that every class will be represented
		with zipfile.ZipFile(self.train_zip) as z:
			train_files = [name for name in z.namelist() if name.endswith(".ppm")]
			random.shuffle(train_files)

		if load_augmented:
			with zipfile.ZipFile(self.get_augmented_name()) as z:
				aug_files = z.namelist()
				random.shuffle(aug_files)
		else:
			aug_files = []

		# avoid creating multiple arrays or copying them by explicitly computing indices and initializing arrays
		n_imgs = sum(class_distr)
		indices = np.array(range(n_imgs))
		imgs = np.zeros((n_imgs, self.img_size, self.img_size, self.n_channels), dtype="float32")
		labels = np.zeros((n_imgs, self.n_classes), dtype="float32")

		if shuffle:
			# shuffle only indices s.t. we can rearrange (image, label) pairs accordingly
			random.seed(self.random_seed)
			random.shuffle(indices)

		def load_data(zipname, filenames, start_index=0):
			"""
			Load data of a given zip file and write it into a shared array in a region determined by start_index.
			"""
			i = start_index
			with zipfile.ZipFile(zipname) as z, warnings.catch_warnings(record=True) as w:
				# ignore UserWarnings from skimage (about future releases)
				warnings.simplefilter('always')
				w = filter(lambda x: issubclass(x.category, UserWarning), w)

				for filename in filenames:
					img_class = self.get_class(filename)
					if n_per_class[img_class] < max_per_class:
						n_per_class[img_class] += 1
					else:
						# skip if we already have enough images of this class
						continue

					with z.open(filename) as f:
						img = io.imread(BytesIO(f.read()))
						img = self.preprocess(img)
						imgs[indices[i]] = img
						labels[indices[i]] = self.get_class(filename, hot_encoded=hot_encoded)

					i += 1
			return i

		# read images and labels from files
		n_loaded = load_data(self.train_zip, train_files)

		# optionally load augmented data
		if load_augmented:
			load_data(self.get_augmented_name(), aug_files, start_index=n_loaded)

		if validation_split:
			return train_test_split(imgs, labels, test_size=validation_split, random_state=self.random_seed)
		else:
			return imgs, labels

	def count_augmented_imgs(self):
		"""Count the number of available images in the augmented dataset."""
		with zipfile.ZipFile(self.get_augmented_name) as z:
			n_imgs = len(z.namelist())
		return n_imgs

	def get_augmented_name(self):
		"""Return the name of the zip file containing the augmented data."""
		return self.train_zip[:-4]+'-augmented.zip'

	def imgs_of_class(self, id):
		"""Return all image filenames in the training set belonging to a certain class."""
		with zipfile.ZipFile(self.train_zip) as z:
			files = filter(lambda x: x.endswith(self.img_extension), z.namelist())
			files = filter(lambda x: self.get_class(x) == id, files)
		return files

	def preprocess(self, img):
		"""Preprocess a given image: scaling to [0, 1], central square cropping and resizing."""
		img = np.asarray(img, dtype="float32") / 255

		# central square crop
		min_side = min(img.shape[:-1])
		centre = img.shape[0] // 2, img.shape[1] // 2
		img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
					centre[1] - min_side // 2:centre[1] + min_side // 2,
					:]
		# resize
		img = transform.resize(img, (self.img_size, self.img_size))

		# channels first
		# img = np.rollaxis(img, -1)
		return np.array(img, dtype="float32")

	def get_class(self, filename, hot_encoded=False):
		"""Return the class id according to a given file path, derived from the containing directory."""
		id = int(filename.split("/")[-2])
		if hot_encoded:
			id = np.eye(self.n_classes)[id]
		return id

	def get_test_data(self, steal=False):
		"""Load the test data from the zip file and return it."""
		X = []
		Y = []
		raise NotImplementedError()
		return X, Y