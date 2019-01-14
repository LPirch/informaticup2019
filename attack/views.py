from project_conf import PROCESS_DIR, IMG_TMP_DIR, ATTACK_PREFIX

from django.shortcuts import render
from django.http import HttpResponse

from utils_proc import is_pid_running, get_token_from_pid, get_running_procs

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
		selected_model = ""
		error_msg = ""

		if "model" in request.GET:
			selected_model = request.GET["model"]

		if 'error' in request.GET:
			error_msg = request.GET['error']

		models = get_models_info(selected=selected_model)

		context = dict(
			{
				'active': 'Attack',
				'models': models,
				'selected_model': selected_model
			},
			**BASE_CONTEXT
		)
		if error_msg:
			context["error"] = error_msg
		return render(request, 'attack/attack.html', context)
	return HttpResponse(status=405)

def overview(request):
	if request.method == "GET":
		processes  = get_running_procs(prefix=ATTACK_PREFIX)

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