from project_conf import PROCESS_DIR, IMG_TMP_DIR

from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from utils_proc import is_pid_running, get_token_from_pid, kill_proc, write_pid

import os
import os.path
import random

from attack.handler import CWL2AttackHandler, RobustCWL2AttackHandler, PhysicalAttackHandler
from models.views import get_models_info

BASE_CONTEXT = {
	'tabs': [
		{'name': 'Overview', 'url': 'overview.html'},
		{'name': 'Attack', 'url': 'attack.html'},
		{'name': 'Details', 'url': 'details.html'}
	]
}

attacks = {
	"cwl2": CWL2AttackHandler,
	"robust_cwl2": RobustCWL2AttackHandler,
	"physical": PhysicalAttackHandler
}

def attack(request):
	models = get_models_info()

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
	try:
		kwargs = attack.parse_arguments(request)
	except Exception as e:
		return HttpResponse("Invalid argument" + str(e))

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

	if "image" in kwargs and kwargs["image"]:
		src_img_path = os.path.join(outdir, "original.png")
		# override image arg with tmp filename of img
		kwargs["image"] = default_storage.save(src_img_path, ContentFile(kwargs["image"].read()))
	
	if "mask_image" in kwargs and kwargs["mask_image"]:
		mask_path = os.path.join(outdir, "mask.png")
		kwargs["mask_image"] = default_storage.save(mask_path, ContentFile(kwargs["mask_image"].read()))

	try:
		pid = attack.start(process_dir, kwargs)
	except Exception as e:
		print(type(e))
		print(str(e))
		return HttpResponse("Error on Popen")

	try:
		write_pid(token, pid)
	except:
		return HttpResponse("Error on create pid")

	return redirect('/attack/details.html?pid=' + str(pid))

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
	if request.method == "POST" and request.POST['pid']:
		kill_proc(int(request.POST['pid']))
		return redirect('/attack/overview.html')

	return HttpResponse(status=400)