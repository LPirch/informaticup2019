from project_conf import PROCESS_DIR, IMG_TMP_DIR

from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

import os
import os.path
import random

from .cwl2 import CWL2AttackHandling

BASE_CONTEXT = {
	'tabs': [
		{'name': 'Overview', 'url': 'overview.html'},
		{'name': 'Attack', 'url': 'attack.html'},
		{'name': 'Details', 'url': 'details.html'}
	]
}

attacks = {
	"cwl2": CWL2AttackHandling
}

def get_token_from_pid(pid):
	pid = str(int(pid))
	pid_dir = os.path.join(PROCESS_DIR, pid)

	if not os.path.exists(pid_dir):
		raise RuntimeError("No process with pid found")

	try:
		with open(pid_dir, "r") as f:
			token = f.read()
	except:
		raise RuntimeError("Could not read from process pid-file")

	return token

def is_pid_running(pid):
	""" Check For the existence of a unix pid. """
	try:
		os.kill(pid, 0)
	except OSError:
		return False
	else:
		return True

def attack(request):
	context = dict(
		{'active': 'Attack'},
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
			'active': 'overview',
			'processes': processes
		}, 
		**BASE_CONTEXT
	)
	return render(request, 'attack/overview.html', context)

def details(request):
	pid = str(int(request.GET["pid"]))
	token = get_token_from_pid(pid)

	if not os.path.exists(os.path.join(PROCESS_DIR, pid)):
		return HttpResponse("No process with pid")

	context = dict(
		{
			'active': 'Details',
			'pid': str(pid),
			'img_path': os.path.join(IMG_TMP_DIR, token)
		},
		**BASE_CONTEXT
	)
	return render(request, 'attack/details.html', context)

def handle_start_attack(request):
	if request.method == "POST":
		attack = request.POST["attack"]

		if attack in attacks:
			return start_attack(request, attacks[attack])

	return HttpResponse("Attack not found")

def start_attack(request, attack):
	try:
		args = attack.handle_arguments(request)
	except:
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

	if args["image"]:
		src_img_path = os.path.join(outdir, "original.png")
		args["src_img_path"] = default_storage.save(src_img_path, ContentFile(args["image"].read()))

	try:
		p = attack.start(outdir=outdir, process_dir=process_dir, **args)
	except Exception as e:
		print(type(e))
		print(str(e))
		return HttpResponse("Error on Popen")

	pid = str(p.pid)

	try:
		with open(os.path.join(PROCESS_DIR, pid), "w") as f:
			f.write(token)
	except:
		return HttpResponse("Error on create pid")

	return redirect('/attack/details.html?pid=' + pid)

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