from project_conf import MODEL_SAVE_PATH, PROCESS_DIR, TRAINING_PREFIX, CACHE_DIR

import os
import pickle
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse

from models.handler import TrainHandler
from utils_proc import gen_token, get_running_procs, get_token_from_pid, write_pid, kill_proc
import subprocess
import re

TRAINING_METHODS = {
	'rebuild': TrainHandler,
	'substitute': TrainHandler
}

ALREADY_TRAINING = "1"
UNKNOWN_TRAINING = "2"

def handle_start_model_info(modelname):
	# avoid starting multiple processes
	if len(get_running_procs(prefix='modelinfo')) > 0:
		# 409 Conflict
		return HttpResponse(status=409)

	# make sure the model actuall exists
	if not os.path.join(MODEL_SAVE_PATH, modelname + '.h5'):
		return HttpResponse(status=404)

	token = gen_token(prefix='modelinfo')
	process_dir = os.path.join(PROCESS_DIR, token)
	os.makedirs(process_dir)
	args = ["python", "start_proc.py", process_dir, "python", "cache_model_arch.py", modelname]
	pid = int(subprocess.check_output(args).strip())
	write_pid(token, pid)

	# Process has been started, inform user:
	# 503 Service Unavailable (i.e. come back later)
	return HttpResponse(status=503)


def handle_model_info(request):
	if request.method == 'GET':
		modelname = request.GET['modelname']

		pkl_path = os.path.join(CACHE_DIR, modelname+'.pkl')

		if not os.path.exists(pkl_path):
			return handle_start_model_info(modelname)

		with open(pkl_path, 'rb') as pkl:
			# stringify shape data for serialization
			layer_info = []
			for layer in pickle.load(pkl):
				layer_dict = {}
				for k,v in layer.items():
					layer_dict[k] = str(v)
				layer_info.append(layer_dict)
		
		procs = get_running_procs(prefix='modelinfo')
		if len(procs) > 0:
			# clean up temporary files of proc
			pid = procs[0]['id']
			kill_proc(pid)

		return JsonResponse({"modelInfo": layer_info})
	return HttpResponse(status=405)


def handle_delete_model(request):
	if request.method == 'POST':
		filename = request.POST['filename']

		# basic sanitizing
		if re.match("^([a-zA-Z0-9_]+\.h5)$", filename):
			return delete_model(filename)
		return HttpResponse(status=400)
	return HttpResponse(status=405)


def handle_upload_model(request):
	if request.method == 'POST':
		if not request.FILES:
			# do nothing if no file is provided
			return HttpResponse(status=400)
		filename = request.FILES['filechooser'].name
		if os.path.exists(os.path.join(MODEL_SAVE_PATH, filename)):
			return HttpResponse(status=400)
		else:
			store_uploaded_file(request.FILES['filechooser'])
		
		return redirect('/models/')
	return HttpResponse(status=405)


def handle_start_training(request):
	if request.method == "POST":
		training = request.POST["training"]
	
		# avoid two concurrent trainings
		if len(get_running_procs(prefix="train")) > 0:
			return redirect('/models/training.html?error=' + ALREADY_TRAINING) 

		if training in TRAINING_METHODS:
			return start_training(request, TRAINING_METHODS[training])

		return redirect('/models/training.html?error=' + UNKNOWN_TRAINING) 
	return HttpResponse(status=405)


def handle_abort_training(request):
	if request.method == 'POST':
		pid = int(request.POST['pid'])
		kill_proc(pid)
		return HttpResponse(status=200)
	return HttpResponse(status=405)


def delete_model(filename):
	filepath = os.path.join(MODEL_SAVE_PATH, filename)
	if os.path.exists(filepath):
		os.remove(filepath)
		return HttpResponse(status=200)
	return HttpResponse(status=404)


def store_uploaded_file(file):
	# create cache directory if not exists
	if not os.path.exists(MODEL_SAVE_PATH):
		os.makedirs(MODEL_SAVE_PATH)
	
	# save file (chunk-wise for handling large files as well)
	with open(os.path.join(MODEL_SAVE_PATH, file.name), 'wb') as f:
		for chunk in file.chunks():
			f.write(chunk)

	modelname = (".").join(file.name.split(".")[:-1])
	invalidate_model_cache(modelname)


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
	
	return redirect('/models/details.html')


def clear_proc(pid):
	token = get_token_from_pid(pid)

	if os.path.exists(os.path.join(PROCESS_DIR, token, 'stdout')):
		os.remove(os.path.join(PROCESS_DIR, token, 'stdout'))
	
	if os.path.exists(os.path.join(PROCESS_DIR, token)):
		os.removedirs(os.path.join(PROCESS_DIR, token))
	
	if os.path.exists(os.path.join(PROCESS_DIR, str(pid))):
		os.remove(os.path.join(PROCESS_DIR, str(pid)))


def invalidate_model_cache(modelname):
	pkl_path = os.path.join(CACHE_DIR, modelname+'.pkl')
	if os.path.exists(pkl_path):
		os.remove(pkl_path)