from django.shortcuts import render, redirect
from django.http import HttpResponse

import os
import os.path
import subprocess
import random

PROCESSES_DIR = ".process/"

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
            "running": os.path.exists("/proc/" + p)
        })

    context = {
        "processes" : processes
    }

    return render(request, 'attack/overview.html', context)

def details(request):
    try:
        pid = str(int(request.GET["pid"]))
    except:
        return HttpResponse("Invaild argument")

    context = {
        "pid" : pid
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
        bss = int(request.POST["binary_search_steps"])
        confidence = int(request.POST["confidence"])
        max_iterations = int(request.POST["max_iterations"])
    except:
        return HttpResponse("Invalid argument")

    if not os.path.exists(PROCESSES_DIR):
        os.makedirs(PROCESSES_DIR)

    token = str(random.random())

    try:
        with open(PROCESSES_DIR + token + ".out", "wb") as f:
            p = subprocess.Popen(['python3', 'list_classes.py'], stdout=f, bufsize=1)
    except:
        return HttpResponse("Error on Popen")

    pid = str(p.pid)

    process_dir = PROCESSES_DIR + pid + "/"

    try:
        os.mkdir(process_dir)
    except:
        return HttpResponse("Error on mkdir")

    try:
        with open(process_dir + "token", "w") as f:
            f.write(token)
    except:
        return HttpResponse("Error on create tokenfile")

    return redirect('/attack/details.html?pid=' + pid)

def handle_proc_info(request):
    try:
        pid = str(int(request.GET["pid"]))
    except:
        return HttpResponse("Invaild argument")

    if not os.path.exists(PROCESSES_DIR + pid) or not os.path.exists("/proc/" + pid):
        return HttpResponse("No process with pid found")

    process_dir = PROCESSES_DIR + pid + "/"

    try:
        with open(process_dir + "token", "r") as f:
            token = f.read()
    except:
        return HttpResponse("Could not read from process pid-file")

    try:
        with open(PROCESSES_DIR + token + ".out", "r") as g:
            out = g.read()
    except:
        return HttpResponse("Could not read process output (" + PROCESSES_DIR + token + ".out)")

    return HttpResponse(out)