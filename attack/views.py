from project_conf import PROCESS_DIR, IMG_TMP_DIR

from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from utils_proc import is_pid_running, get_token_from_pid, kill_proc

import os
import os.path
import random

from attack.handler import CWL2AttackHandler
from models.views import get_models_info

BASE_CONTEXT = {
	'tabs': [
		{'name': 'Overview', 'url': 'overview.html'},
		{'name': 'Attack', 'url': 'attack.html'},
		{'name': 'Details', 'url': 'details.html'}
	]
}

attacks = {
	"cwl2": CWL2AttackHandler
}

def attack(request):
	selected_model = request.session.get('selected_model')
	models = get_models_info(selected_model)

	context = dict(
		{
			'active': 'Attack',
			'models': models
		},
		**BASE_CONTEXT
	)
	return render(request, 'attack/attack.html', context)

def overview(request):
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

def details(request):
	pid = request.GET.get('pid', '')

	context = dict({'active': 'Details'}, **BASE_CONTEXT)
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

def handle_start_attack(request):
	if request.method == "POST":
		attack = request.POST["attack"]

		if attack in attacks:
			return start_attack(request, attacks[attack])

	return HttpResponse("Attack not found")

def start_attack(request, attack):
	import traceback
	try:
		kwargs = attack.parse_arguments(request)
	except:
		traceback.print_exc()
		return HttpResponse("Invalid argument")

	if not os.path.exists(PROCESS_DIR):
		os.makedirs(PROCESS_DIR)

	token = str(random.random())
	process_dir = os.path.join(PROCESS_DIR, token)

	try:
		os.mkdir(process_dir)
	except:
		return HttpResponse("Error on mkdir")

	outdir = os.path.join(IMG_TMP_DIR, token)
	kwargs["outdir"] = outdir

	if kwargs["image"]:
		src_img_path = os.path.join(outdir, "original.png")
		# override image arg with tmp filename of img
		kwargs["image"] = default_storage.save(src_img_path, ContentFile(kwargs["image"].read()))

	try:
		pid = attack.start(process_dir, kwargs)
	except Exception as e:
		print(type(e))
		print(str(e))
		traceback.print_exc()
		return HttpResponse("Error on Popen")

	try:
		with open(os.path.join(PROCESS_DIR, str(pid)), "w") as f:
			f.write(token)
	except:
		return HttpResponse("Error on create pid")

	return redirect('/attack/overview.html')

def handle_proc_info(request):
	token = get_token_from_pid(request.GET["pid"])
	process_dir = os.path.join(PROCESS_DIR, token)

	try:
		with open(os.path.join(process_dir, "stdout"), "r") as g:
			out = g.read()
	except:
		return HttpResponse("Could not read process output (" + PROCESS_DIR + token + ".out)")

	return HttpResponse(out)

def handle_list_images(request):
	token = get_token_from_pid(request.GET["pid"])
	process_dir = os.path.join(IMG_TMP_DIR, token)

	try:
		images = list(filter(lambda x: x.endswith(".png"), os.listdir(process_dir)))
	except:
		return JsonResponse({"images": []})

	return JsonResponse({"images": images})

def handle_delete_proc(request):
	pid = request.GET.get('pid', '')

	if pid:
		pid = int(pid)
		kill_proc(pid)

		return redirect('/attack/overview.html')
	else:
		return HttpResponse(status=400)