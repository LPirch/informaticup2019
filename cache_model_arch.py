from project_conf import CACHE_DIR, MODEL_SAVE_PATH
import pickle
import os
import sys

from keras.models import load_model

if __name__ == '__main__':
	"""Usage: python cache_model_arch.py [modelname]
		Caches a model's architecture in a pickle file.
	"""
	if not os.path.exists(CACHE_DIR):
		os.makedirs(CACHE_DIR)
	
	modelname = sys.argv[1]

	model_path = os.path.join(MODEL_SAVE_PATH, modelname+'.h5')
	pkl_path = os.path.join(CACHE_DIR, modelname+'.pkl')

	if not os.path.exists(pkl_path):
		model = load_model(model_path)
		layer_info = []
		for layer in model.layers:
			layer_info.append({
				'name': layer.name,
				'input_shape': layer.input_shape,
				'output_shape': layer.output_shape
			})
		
		with open(pkl_path, 'wb') as pkl:
			pickle.dump(layer_info, pkl)
