from project_conf import MODEL_SAVE_PATH, PROCESS_DIR, TRAINING_PREFIX, CACHE_DIR

import os
import json
import pickle
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django import forms
from time import strftime, ctime
from datetime import datetime

from models.train import train_rebuild
from models.handler import TrainHandler
from utils_proc import gen_token, get_running_procs, get_token_from_pid, is_pid_running, write_pid, kill_proc
import subprocess

BASE_CONTEXT = {
	'tabs': [
		{'name': 'Overview', 'url': 'overview.html'},
		{'name': 'Training', 'url': 'training.html'},
		{'name': 'Details', 'url': 'details.html'}
	]
}

TRAINING_METHODS = {
	'rebuild': TrainHandler,
	'substitute': TrainHandler
}

def overview(request):
	selected_model = request.session.get('selected_model')
	models = get_models_info(selected_model)

	context = dict(
		{
			'active': 'Overview',
			'models': models
		}, 
		**BASE_CONTEXT
	)
	return render(request, 'models/overview.html', context)

def get_models_info(selected_model = ''):
	models = []
	for _, _, filenames in os.walk(MODEL_SAVE_PATH):
		for f in filenames:
			filepath = os.path.join(MODEL_SAVE_PATH, f)
			last_modified = datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%Y-%d-%m')
			models.append({
				'name': f, 
				'modified': last_modified,
				'selected': f == selected_model,
				'size': readable_file_size(os.path.getsize(filepath))
			})
	return models

def readable_file_size(n_bytes, suffix='B'):
	for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
		if abs(n_bytes) < 1e3:
			return "%3.1f%s%s" % (n_bytes, unit, suffix)
		n_bytes /= 1e3
	return "%.1f%s%s" % (n_bytes, 'Y', suffix)

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

def handle_upload_model(request):
	if request.method == 'POST':
		if not request.FILES:
			# do nothing if no file is provided
			return HttpResponse()
		filename = request.FILES['filechooser'].name
		if os.path.exists(os.path.join(MODEL_SAVE_PATH, filename)):
			return HttpResponse(400)
		else:
			store_uploaded_file(request.FILES['filechooser'])
		
		# reload model table after file upload
		selected_model = request.session.get('selected_model')
		models = get_models_info(selected_model)

		return overview(request)
	return HttpResponse(400)

def store_uploaded_file(file):
	# create cache directory if not exists
	if not os.path.exists(MODEL_SAVE_PATH):
		os.makedirs(MODEL_SAVE_PATH)
	
	# save file (chunk-wise for handling large files as well)
	with open(os.path.join(MODEL_SAVE_PATH, file.name), 'wb') as f:
		for chunk in file.chunks():
			f.write(chunk)

	modelname = file.name.split(".")[:-1].join(".")
	invalidate_model_cache(modelname)

def training(request):
	context = dict(
		{'active': 'Training'},
		**BASE_CONTEXT
	)
	return render(request, 'models/training.html', context)

def handle_start_training(request):
	if request.method == "POST":
		training = request.POST["training"]
	
	# avoid two concurrent trainings
	if len(get_running_procs(prefix="train")) > 0:
		context = dict(
			{
				'active': 'Training',
				'error_unavailable': True
			}, 
			**BASE_CONTEXT
		)
		return render(request, 'models/training.html', context) 

	if training in TRAINING_METHODS:
		return start_training(request, TRAINING_METHODS[training])

	return HttpResponse("Training method not found")

def start_training(request, training):
	try:
		kwargs = training.parse_arguments(request)
		invalidate_model_cache(kwargs['modelname'])
	except Exception as e:
		print(type(e))
		print(str(e))
		return HttpResponse("Invalid argument")
	
	token = gen_token(TRAINING_PREFIX)
	process_dir = os.path.join(PROCESS_DIR, token)

	try:
		os.mkdir(process_dir)
	except:
		return HttpResponse("Error on mkdir")
	
	try:
		pid = training.start(process_dir, kwargs)
	except Exception as e:
		print(type(e))
		print(str(e))
		return HttpResponse("Error on Popen")

	try:
		write_pid(token, pid)
	except:
		return HttpResponse("Error on create pid")
	
	context = dict(
			{
				'active': 'Details',
				'pid': str(pid)
			}, 
			**BASE_CONTEXT
		)
	return render(request, 'models/details.html', context)

def handle_abort_training(request):
	if request.method == 'GET':
		pid = int(request.GET.get('pid', ''))
		kill_proc(pid)
	return HttpResponse()

def clear_proc(pid):
	token = get_token_from_pid(pid)

	if os.path.exists(os.path.join(PROCESS_DIR, token, 'stdout')):
		os.remove(os.path.join(PROCESS_DIR, token, 'stdout'))
	
	if os.path.exists(os.path.join(PROCESS_DIR, token)):
		os.removedirs(os.path.join(PROCESS_DIR, token))
	
	if os.path.exists(os.path.join(PROCESS_DIR, str(pid))):
		os.remove(os.path.join(PROCESS_DIR, str(pid)))

def details(request):
	context = dict(
		{'active': 'Details'},
		**BASE_CONTEXT
	)
	procs = get_running_procs(prefix="train")
	if len(procs) == 0:
		context['error_none_running'] = True
	else:
		context.update({
			'pid': procs[0]['id']
		})

	return render(request, 'models/details.html', context)

# TODO init MODELSPECS_DIR (train.py), show architecture
"""
def get_model_summary(modelname):
	model = load_model(os.path.join(MODELSPEC_PATH, modelname))
	K.clear_session()
	layer_info = []
	for layer in model.layers:
		layer_info.append({
			'name': layer.name,
			'input_shape': layer.input_shape,
			'output_shape': layer.output_shape
		})
	
	return layer_info
"""

def handle_get_selected(request):
	selected_model = request.session.get('selected_model')
	return JsonResponse(selected_model, safe=False)


def handle_start_model_info(request):
	if request.method == 'GET' and 'modelname' in request.GET:
		# avoid starting multiple processes
		if len(get_running_procs(prefix='modelinfo')) > 0:
			return HttpResponse()

		modelname = request.GET['modelname']
		token = gen_token(prefix='modelinfo')
		process_dir = os.path.join(PROCESS_DIR, token)
		os.makedirs(process_dir)
		args = ["python", "start_proc.py", process_dir, "python", "cache_model_arch.py", modelname]
		pid = int(subprocess.check_output(args).strip())
		write_pid(token, pid)
		return HttpResponse()

	return HttpResponse(status=400)


def handle_model_info(request):
	if request.method == 'POST':
		try:
			data = json.loads(request.body.decode('utf-8'))
			modelname = data['modelname']
		except:
			return HttpResponse(status=404)
		pkl_path = os.path.join(CACHE_DIR, modelname+'.pkl')
		if os.path.exists(pkl_path):
			with open(pkl_path, 'rb') as pkl:
				# stringify shape data for serialization
				layer_info = []
				for layer in pickle.load(pkl):
					layer_dict = {}
					for k,v in layer.items():
						layer_dict[k] = str(v)
					layer_info.append(layer_dict)
		else:
			return HttpResponse(status=404)
		
		procs = get_running_procs(prefix='modelinfo')
		if len(procs) > 0:
			# clean up temporary files of proc
			pid = procs[0]['id']
			kill_proc(pid)

		return JsonResponse(layer_info, safe=False)
		
	return HttpResponse(status=400)

def invalidate_model_cache(modelname):
	pkl_path = os.path.join(CACHE_DIR, modelname+'.pkl')
	if os.path.exists(pkl_path):
		os.remove(pkl_path)
"""
training_methods = {
	'rebuild': TrainHandler,
	'substitute': TrainHandler
}

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
	
	return render(request, 'models/overview.html', context)

def models(request):
	selected_model = request.session.get('selected_model')
	context = { 
		"train" : {"active_class": "active"},
		"selected_model": "None"
	}
	
	return render(request, 'models/models.html')

def training(request):
	context = {}
	try:
		procs = get_running_procs(prefix=TRAINING_PREFIX)
		if len(procs) > 0:
			context.update({
				'pid': procs[0]['id']
			})
	except ValueError as e:
		print("WARNING: ignored the following ValueError: "+str(e))
	
	return render(request, 'models/training.html', context)

def tensorboard(request):    
	return render(request, 'models/tensorboard.html')

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
			return redirect('/models/models.html', context)
		filename = request.FILES['filechooser'].name
		if os.path.exists(os.path.join(MODEL_SAVE_PATH, filename)):
			return HttpResponse(400)
		else:
			store_uploaded_file(request.FILES['filechooser'])
		
		# reload model table after file upload
		context = { "train" : {"active_class": "active"} }
		return redirect('/models/models.html', context)
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
	model = load_model(os.path.join(MODEL_SAVE_PATH, modelname))
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
	if request.method == "POST":
		training = request.POST["training"]
	
	if training in training_methods:
		return start_training(request, training_methods[training])

	return HttpResponse("Training method not found")

def handle_proc_info(request):
	status = {}
	if request.method == "GET":
		pid = request.GET.get('pid', '')
		if pid and is_pid_running(pid):
			status = {'running': True}
		else:
			status = {'running': False}

	return JsonResponse(status, safe=False)

def start_training(request, training):
	try:
		kwargs = training.parse_arguments(request)
	except Exception as e:
		print(type(e))
		print(str(e))
		return HttpResponse("Invalid argument")
	
	token = gen_token(TRAINING_PREFIX)
	process_dir = os.path.join(PROCESS_DIR, token)

	try:
		os.mkdir(process_dir)
	except:
		return HttpResponse("Error on mkdir")
	
	try:
		pid = training.start(process_dir, kwargs)
	except Exception as e:
		print(type(e))
		print(str(e))
		return HttpResponse("Error on Popen")

	try:
		write_pid(token, pid)
	except:
		return HttpResponse("Error on create pid")
	
	context = {
		'pid': pid
	}
	return render(request, 'models/training.html', context)

def handle_proc_delete(request):
	pid = request.GET.get('pid')
	clear_proc(pid)
	context = { "train" : {"active_class": "active"} }
	return redirect('/models/models.html', context)

def clear_proc(pid):
	token = get_token_from_pid(pid)

	if os.path.exists(os.path.join(PROCESS_DIR, token, 'stdout')):
		os.remove(os.path.join(PROCESS_DIR, token, 'stdout'))
	
	print(os.path.join(PROCESS_DIR, token))
	if os.path.exists(os.path.join(PROCESS_DIR, token)):
		os.removedirs(os.path.join(PROCESS_DIR, token))
	
	print(os.path.join(PROCESS_DIR, str(pid)))
	if os.path.exists(os.path.join(PROCESS_DIR, str(pid))):
		os.remove(os.path.join(PROCESS_DIR, str(pid)))
"""