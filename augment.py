import os
import zipfile
import numpy as np
from skimage import io
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import warp, AffineTransform
from keras.preprocessing.image import ImageDataGenerator

from gtsrb import GTSRB


def augment_dataset(dataset, target_count, datagen_kwargs):
	distr = dataset.get_class_distr(dataset)
	if target_count < 0:
		target_count = np.max(distr)

	# number of data points to generate per class
	n_to_generate = np.ones(dataset.n_classes) * target_count - distr
	# may not be negative
	n_to_generate = n_to_generate.astype(int).clip(min=0)
	
	# initialize generator
	datagen = ImageDataGenerator(**datagen_kwargs)

	out_zip = dataset.get_augmented_name()
	if os.path.exists(out_zip):
		os.remove(out_zip)
	for curr_class in range(dataset.n_classes):
		if n_to_generate[curr_class] == 0:
			continue

		img_names = list(dataset.imgs_of_class(curr_class))
		# pick n source samples with replacement
		img_names = np.random.choice(img_names, size=n_to_generate[curr_class], replace=True)

		imgs = []
		with zipfile.ZipFile(dataset.train_zip) as z_in:
			for filename in img_names:
				with z_in.open(filename) as f:
					img = io.imread(BytesIO(f.read()))
					img = dataset.preprocess(img)
					imgs.append(img)

		imgs = np.array(imgs)
		labels = np.ones(len(imgs)) * curr_class

		with zipfile.ZipFile(out_zip, 'a', zipfile.ZIP_DEFLATED, False) as z_out:
			print("#"*80)
			print(imgs.shape, labels.shape)
			for mod_imgs, _ in datagen.flow(imgs, labels, batch_size=len(imgs)):
				for i, img in enumerate(mod_imgs):
					#mod_filename = os.path.join(str(label), str(i)+'.ppm')
					img = np.rint(img * 255).astype('uint8')
					mod_filename = str(i)+".ppm"
					img = Image.fromarray(img, 'RGB')
					dirname = "tmp/"+str(curr_class)
					# TODO: is there a better way than saving a temporary file and archiving it?
					os.makedirs(dirname)
					img.save(dirname+"/"+mod_filename)
					z_out.write(dirname+"/"+mod_filename, arcname=str(curr_class)+"/"+mod_filename)
					os.remove(dirname+"/"+mod_filename)
					os.rmdir(dirname)
				break
			



def main():
	random_seed = 42
	datagen_kwargs = {
		'featurewise_center': False,
		'samplewise_center': False,
		'featurewise_std_normalization': False,
		'samplewise_std_normalization': False,
		'zoom_range': 0.2,
		'rotation_range': 30,
		'shear_range': 0.2,
		'width_shift_range':0.2,
		'height_shift_range': 0.2,
		'channel_shift_range': 0
	}
	np.random.seed(random_seed)
	gtsrb = GTSRB('data', random_seed)
	distr = gtsrb.get_class_distr(gtsrb, load_augmented=False)
	gtsrb.plot_class_distr(distr, output_file="before.png")
	
	augment_dataset(gtsrb, -1, datagen_kwargs)

	distr = gtsrb.get_class_distr(gtsrb, load_augmented=True)
	gtsrb.plot_class_distr(distr, output_file="after.png")


	"""
	with zipfile.ZipFile("data/GTSRB_Final_Training_Images.zip") as z:
		files = [name for name in z.namelist() if name.endswith(".ppm")]
		files = files[:100:10]
		for i,name in enumerate(files):
			print(name)
			with z.open(name) as f:
				img = io.imread(BytesIO(f.read())) / 255
				label = 0
				datagen.fit([img])
				print(type(img), img.shape, np.min(img), np.max(img)) 
				img_new, label = datagen.flow(np.array([img]), np.array([label]), batch_size=1).next()
				print(type(img_new), img_new.shape, np.min(img_new), np.max(img_new))
				img_new = img_new.reshape(img_new.shape[1:])
				
				plt.subplot(121)
				plt.imshow(img)
				plt.subplot(122)
				plt.imshow(img_new)
				plt.savefig("tmp/fig-"+str(i)+".png")
	"""

if __name__ == '__main__':
	main()