from project_conf import PROCESS_DIR, IMG_TMP_DIR, CACHE_DIR, API_KEY_LOCATION, REMOTE_URL

from django.shortcuts import redirect
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from utils_proc import is_pid_running, get_token_from_pid, kill_proc, write_pid
from attack.handler import CWL2AttackHandler, RobustCWL2AttackHandler, PhysicalAttackHandler

from PIL import Image
from gtsrb import GTSRB

import os
import os.path
import random
import json
import requests
import pickle
import re

attacks = {
	"cwl2": CWL2AttackHandler,
	"robust_cwl2": RobustCWL2AttackHandler,
	"physical": PhysicalAttackHandler
}

def handle_proc_info(request):
	if request.method == "GET":
		pid = str(int(request.GET["pid"]))
		token = get_token_from_pid(pid)
		process_dir = os.path.join(PROCESS_DIR, token)

		try:
			with open(os.path.join(process_dir, "stdout"), "r") as g:
				out = g.read()
		except:
			return HttpResponse("Could not read process output (" + PROCESS_DIR + token + ".out)")

		return JsonResponse({"console": out, "running": is_pid_running(pid)})
	return HttpResponse(status=405)


def handle_list_images(request):
	if request.method == "GET":
		pid = str(int(request.GET["pid"]))
		token = get_token_from_pid(pid)
		process_dir = os.path.join(IMG_TMP_DIR, token)

		try:
			images = list(filter(lambda x: x.endswith(".png"), os.listdir(process_dir)))
		except:
			return HttpResponse(status=400)

		return JsonResponse({"images": images, "running": is_pid_running(pid)})
	return HttpResponse(status=405)


def handle_classify(request):
	if request.method == "GET":
		image = request.GET["image"]
		token = get_token_from_pid(request.GET["pid"])

		if not re.match("^[a-zA-Z0-9_]+\.png$", image):
			return HttpResponse(status=400)

		process_dir = os.path.join(IMG_TMP_DIR, token)
		img_path = os.path.join(process_dir, image)

		if not os.path.exists(img_path):
			return HttpResponse(status=404)

		cache_dir = os.path.join(CACHE_DIR, token)
		cache_file = os.path.join(cache_dir, image + ".cache")

		if not os.path.exists(cache_dir):
			os.makedirs(cache_dir)

		if os.path.exists(cache_file):
			with open(cache_file, "rb") as f:
				remote = pickle.load(f)
		else:
			with open(img_path, "rb") as f:
				remote = fetch_oracle_response(f.read())

			try:
				with open(cache_file, "wb") as f:
					pickle.dump(remote, f)
			except:
				print("Creating pickle failed", cache_file)

		return JsonResponse({"remote": remote})
	return HttpResponse(status=405)

def load_api_key(loc=API_KEY_LOCATION):
	with open(loc, 'r') as f:
		key = f.read().strip()
	return key

def fetch_oracle_response(img):
	api_key = load_api_key()

	r = requests.post(REMOTE_URL, data={'key': api_key}, files={'image': img})

	if r.status_code != 200:
		log_http_error(r.status_code, r.text)
		raise RuntimeError

	return json.loads(r.text)

def handle_start_attack(request):
	if request.method == "POST":
		attack = request.POST["attack"]

		if attack in attacks:
			return start_attack(request, attacks[attack])
		return HttpResponse(status=400)
	return HttpResponse(status=405)


def handle_delete_proc(request):
	if request.method == "POST":
		kill_proc(int(request.POST['pid']))
		return redirect('/attack/overview.html')
	return HttpResponse(status=405)


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

	try:
		if "image" in kwargs and kwargs["image"]:
			src_img_path = os.path.join(outdir, "original.png")
			# override image arg with tmp filename of img
			img = ContentFile(kwargs["image"].read())
			validate_img_size(img.file, identifier='Source')
			kwargs["image"] = default_storage.save(src_img_path, img)
		
		if "mask_image" in kwargs and kwargs["mask_image"]:
			mask_path = os.path.join(outdir, "mask.png")
			mask_img = ContentFile(kwargs["mask_image"].read())
			validate_img_size(mask_img.file, identifier='Mask')
			kwargs["mask_image"] = default_storage.save(mask_path, mask_img)
	except ValueError as e:
		return redirect('/attack/attack.html?model='+kwargs['model']+'&error='+str(e))

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


def validate_img_size(img, identifier=None):
	img = Image.open(img)
	dataset = GTSRB(random_seed=42)
	expected = (dataset.img_size, dataset.img_size)

	if img.size != expected:
		if not identifier:
			identifier="Input"
		raise ValueError(identifier+" image size mismatch, was "+str(img.size)+ " but expected "+str(expected))
