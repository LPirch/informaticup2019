from project_conf import PROCESS_DIR, IMG_TMP_DIR

from django.shortcuts import render
from django.http import HttpResponse

from utils_proc import is_pid_running, get_token_from_pid

import os
import os.path
import random

from models.views import get_models_info

BASE_CONTEXT = {
	'tabs': [
		{'name': 'Overview', 'url': 'overview.html'},
		{'name': 'Attack', 'url': 'attack.html'}
	]
}

def attack(request):
	if request.method == "GET":
		models = get_models_info()

		context = dict(
			{
				'active': 'Attack',
				'models': models
			},
			**BASE_CONTEXT
		)
		return render(request, 'attack/attack.html', context)
	return HttpResponse(status=405)

def overview(request):
	if request.method == "GET":
		processes  = []

		for p in filter(lambda x: x.isdigit(), os.listdir(PROCESS_DIR)):
			processes.append({
				"id": p,
				"running": is_pid_running(int(p))
			})

		context = dict(
			{
				'active': 'Overview',
				'processes': processes
			}, 
			**BASE_CONTEXT
		)
		return render(request, 'attack/overview.html', context)
	return HttpResponse(status=405)

def details(request):
	if request.method == "GET":
		pid = request.GET['pid']

		context = dict({'active': 'Details'}, **BASE_CONTEXT)
		context["tabs"] = [_ for _ in context["tabs"]]
		context["tabs"].append({'name': 'Details', 'url': 'details.html'})

		if pid:
			pid = int(pid)
			token = get_token_from_pid(pid)
			context.update({
				'pid': str(pid),
				'img_path': os.path.join(IMG_TMP_DIR, token)
			})
		else:
			context['error_none_selected'] =  True

		return render(request, 'attack/details.html', context)
	return HttpResponse(status=405)