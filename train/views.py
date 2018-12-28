import os
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django import forms
from time import strftime, ctime
from datetime import datetime
from keras.models import load_model
import keras.backend as K

from train.train import train_rebuild
from utils_proc import gen_token, get_running_procs, get_token_from_pid, is_pid_running, write_pid
import subprocess

MODEL_SAVE_PATH = os.path.join('data', 'models')
PROCESS_DIR = ".process"

def overview(request):
	selected_model = request.session.get('selected_model')
	context = { 
		"train" : {"active_class": "active"},
		"selected_model": "None"
	}
	if selected_model:
		model_info = get_model_summary(selected_model)
		context['layers'] = model_info
		context['selected_model'] = selected_model
	
	return render(request, 'train/overview.html', context)

def models(request):
	selected_model = request.session.get('selected_model')
	context = { 
		"train" : {"active_class": "active"},
		"selected_model": "None"
	}
	
	return render(request, 'train/models.html')

def training(request):
	return render(request, 'train/training.html')

def tensorboard(request):    
	return render(request, 'train/tensorboard.html')

def handle_model_reload(request):
	selected_model = request.session.get('selected_model')
	models = get_models_info(selected_model)
	return JsonResponse(models, safe=False)

def get_models_info(selected_model = ''):
	files = []
	for _, _, filenames in os.walk(MODEL_SAVE_PATH):
		for f in filenames:
			filepath = os.path.join(MODEL_SAVE_PATH, f)
			last_modified = datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%Y-%d-%m')
			files.append({
				'name': f, 
				'modified': last_modified,
				'selected': f == selected_model
			})
	return files

def handle_uploaded_file(request):
	if request.method == 'POST':
		if not request.FILES:
			# should be prevented by JS, but reload page as double-check
			context = { "train" : {"active_class": "active"} }
			return redirect('/train/models.html', context)
		filename = request.FILES['filechooser'].name
		if os.path.exists(os.path.join(MODEL_SAVE_PATH, filename)):
			return HttpResponse(400)
		else:
			store_uploaded_file(request.FILES['filechooser'])
		
		# reload model table after file upload
		context = { "train" : {"active_class": "active"} }
		return redirect('/train/models.html', context)
	return HttpResponse(400)

def store_uploaded_file(file):
	# create cache directory if not exists
	if not os.path.exists(MODEL_SAVE_PATH):
		os.makedirs(MODEL_SAVE_PATH)
	
	# save file (chunk-wise for handling large files as well)
	with open(os.path.join(MODEL_SAVE_PATH, file.name), 'wb') as f:
		for chunk in file.chunks():
			f.write(chunk)

def handle_delete_model(request):
	if request.method == 'GET':
		filename = request.GET.get('filename', '')
		delete_model(filename)
		if filename == request.session.get('selected_model'):
			request.session['selected_model'] = None
	return HttpResponse()

def delete_model(filename):
	filepath = os.path.join(MODEL_SAVE_PATH, filename)
	if os.path.exists(filepath):
		os.remove(filepath)

def handle_select_model(request):
	if request.method == 'GET':
		filename = request.GET.get('filename', '')
		request.session['selected_model'] = filename
	return HttpResponse()

def get_model_summary(modelname):
	model = load_model(os.path.join('model', 'trained', modelname))
	K.clear_session()
	layer_info = []
	for layer in model.layers:
		layer_info.append({
			'name': layer.name,
			'input_shape': layer.input_shape,
			'output_shape': layer.output_shape
		})
	
	return layer_info

def handle_start_training(request):
	train_method, popen_args = parse_train_params(request)
	
	token = gen_token()
	process_dir = os.path.join(PROCESS_DIR, token)
	print(">>", token)

	try:
		os.makedirs(process_dir)
	except:
		return HttpResponse("Error during mkdir")
	
	try:
		with open(os.path.join(process_dir, 'stdout'), 'wb') as f:
			p = subprocess.Popen(["python", "train_model.py", 
								*popen_args], stdout=f, stderr=f, 
								bufsize=1, universal_newlines=True)
	except Exception as e:
		print(type(e))
		print(str(e))
		return HttpResponse("Error during Popen")
	
	pid = str(p.pid)
	try:
		write_pid(token, pid)
	except Exception as e:
		print(type(e))
		print(str(e))
		return HttpResponse("Error during create PID")


	return redirect('/train/training.html?pid='+pid)

def parse_train_params(request):
	if request.POST["training"] == "rebuild":
		train_method = train_rebuild
	elif request.POST["training"] == "substitute":
		train_method = train_substitute

	popen_args = [
			'--modelname', str(request.POST["modelname"]),
			'--epochs', str(request.POST["epochs"]),
			'--batch_size', str(request.POST["batch_size"]),
			'--learning_rate', str(request.POST["lr"]),
			'--optimizer', str(request.POST["optimizer"]),
			'--dataset', str(request.POST["dataset"]),
			'--validation_split', str(request.POST["valsplit"]),
			'--max_per_class', str(request.POST["maxperclass"])
	]
	
	if int(request.POST["augmentation"]):
		popen_args.append('--load_augmented')
	if int(request.POST["tensorboard"]):
		popen_args.append('--enable_tensorboard')

	return train_method, popen_args
	
	
	
"""
def start_rebuild(request):
	try:
		training = train_rebuild
		train_dict = {
			'modelname': str(request.POST["modelname"]),
			'epochs': int(request.POST["epochs"]),
			'batch_size': int(request.POST["batch_size"]),
			'learning_rate': float(request.POST["lr"]),
			'optimizer': str(request.POST["optimizer"]),
			'dataset': str(request.POST["dataset"]),
			'validation_split': float(request.POST["valsplit"]),
			'max_per_class': int(request.POST["maxperclass"]),
			'load_augmented': bool(request.POST["augmentation"]),
			'enable_tensorboard': bool(request.POST["tensorboard"])
		}

		#TODO actually start training
	except Exception as e:
		return HttpResponse("Invalid argument")
	
	context = {
		'training_running': True
	}
	return render(request, 'train/training.html', context)
"""