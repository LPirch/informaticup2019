from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse

import os
import os.path
import subprocess
import random

PROCESSES_DIR = ".process/"

def get_token_from_pid(pid):
    pid = str(int(pid))

    if not os.path.exists(PROCESSES_DIR + pid):
        raise RuntimeError("No process with pid found")

    try:
        with open(PROCESSES_DIR + pid, "r") as f:
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

def index(request):
    context = {
        "attack" : {"active_class": "active"}
    }
    return render(request, 'attack/overview.html', context)

def attack(request):
    return render(request, 'attack/attack.html')

def overview(request):
    processes  = []

    for p in filter(lambda x: x.isdigit(), os.listdir(PROCESSES_DIR)):
        processes.append({
            "id": p,
            "running": is_pid_running(int(p))
        })

    context = {
        "processes" : processes
    }

    return render(request, 'attack/overview.html', context)

def details(request):
    pid = str(int(request.GET["pid"]))
    token = get_token_from_pid(pid)

    context = {
        "pid" : pid,
        "img_path": "/static/img/" + token + "/"
    }

    if not os.path.exists(PROCESSES_DIR + pid):
        return HttpResponse("No process with pid")

    return render(request, 'attack/details.html', context)

def handle_start_attack(request):
    if request.method == "POST":
        if request.POST["attack"] == "cwl2":
            return start_cwl2(request)

    return HttpResponse("Attack not found")

def start_cwl2(request):
    try:
        attack = "cwl2"
        bss = str(int(request.POST["binary_search_steps"]))
        confidence = str(int(request.POST["confidence"]))
        max_iterations = str(int(request.POST["max_iterations"]))
        target = str(int(request.POST["target"]))
    except:
        return HttpResponse("Invalid argument")

    if not os.path.exists(PROCESSES_DIR):
        os.makedirs(PROCESSES_DIR)

    token = str(random.random())
    process_dir = PROCESSES_DIR + token + "/"

    try:
        os.mkdir(process_dir)
    except:
        return HttpResponse("Error on mkdir")

    try:
        with open(process_dir + "stdout", "wb") as f:
            p = subprocess.Popen(["python3", "attack_model.py",
                "--attack", "cwl2",
                "--model", "gtsrb_model",
                "--model_folder", "model/trained/",
                "--outdir", "static/img/" + token + "/",
                "--binary_search_steps", bss,
                "--confidence", confidence,
                "--max_iterations", max_iterations,
                "--target", target,
                "--image", "gi.png"], stdout=f, stderr=f, bufsize=1, universal_newlines=True)

    except Exception as e:
        print(type(e))
        print(str(e))
        return HttpResponse("Error on Popen")

    pid = str(p.pid)

    try:
        with open(PROCESSES_DIR + pid, "w") as f:
            f.write(token)
    except:
        return HttpResponse("Error on create pid")

    return redirect('/attack/details.html?pid=' + pid)

def handle_proc_info(request):
    token = get_token_from_pid(request.GET["pid"])
    process_dir = PROCESSES_DIR + token + "/"

    try:
        with open(process_dir + "stdout", "r") as g:
            out = g.read()
    except:
        return HttpResponse("Could not read process output (" + PROCESSES_DIR + token + ".out)")

    return HttpResponse(out)

def handle_list_images(request):
    token = get_token_from_pid(request.GET["pid"])
    process_dir = "static/img/" + token + "/"

    images = list(filter(lambda x: x.endswith(".png"), os.listdir(process_dir)))

    return JsonResponse({"images": images})