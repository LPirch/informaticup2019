from project_conf import MODEL_SAVE_PATH, PROCESS_DIR, TRAINING_PREFIX, CACHE_DIR

import os
from django.shortcuts import render
from django.http import HttpResponse
from time import strftime, ctime
from datetime import datetime

from utils_proc import get_running_procs
from models.rest import ALREADY_TRAINING, UNKNOWN_TRAINING

BASE_CONTEXT = {
	'tabs': [
		{'name': 'Overview', 'url': 'overview.html'},
		{'name': 'Training', 'url': 'training.html'},
		{'name': 'Details', 'url': 'details.html'}
	]
}

def overview(request):
	if request.method == "GET":
		models = get_models_info()

		context = dict(
			{
				'active': 'Overview',
				'models': models
			}, 
			**BASE_CONTEXT
		)
		return render(request, 'models/overview.html', context)
	return HttpResponse(status=405)


def get_models_info(selected=None):
	models = []
	for _, _, filenames in os.walk(MODEL_SAVE_PATH):
		for f in filenames:
			filepath = os.path.join(MODEL_SAVE_PATH, f)
			last_modified = datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%Y-%d-%m')
			models.append({
				'name': f, 
				'modified': last_modified,
				'selected': f == selected,
				'size': readable_file_size(os.path.getsize(filepath))
			})
	return models


def readable_file_size(n_bytes, suffix='B'):
	for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
		if abs(n_bytes) < 1e3:
			return "%3.1f%s%s" % (n_bytes, unit, suffix)
		n_bytes /= 1e3
	return "%.1f%s%s" % (n_bytes, 'Y', suffix)


def training(request):
	if request.method == "GET":
		context = dict(
			{'active': 'Training'},
			**BASE_CONTEXT
		)

		if 'error' in request.GET:
			if request.GET['error'] == ALREADY_TRAINING:
				error_msg = 'Training has not been started because concurrent training processes are not supported.'
			elif request.GET['error'] == UNKNOWN_TRAINING:
				error_msg = 'The provided training method is unknown.'
			else:
				error_msg = 'Unexpected error occured.'
			context.update({'error_msg': error_msg})
		
		return render(request, 'models/training.html', context)
	return HttpResponse(status=405)


def details(request):
	if request.method == "GET":
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
	return HttpResponse(status=405)