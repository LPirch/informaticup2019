from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation

from keras.models import Sequential


def cnn_model(img_size, n_classes):
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
	model.add(Dense(n_classes))
	model.add(Activation('softmax'))

	return model

def dense_model(img_size, n_classes):
	model = Sequential()

	model.add(Flatten())
	model.add(Dense(img_size * img_size * 3, activation='relu'))
	model.add(Dropout(0.05))
	model.add(Dense(img_size * img_size, activation='relu'))
	model.add(Dense(n_classes))
	model.add(Activation('softmax'))

	return model