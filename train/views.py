from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django import forms
from time import strftime, ctime
from datetime import datetime
from os import walk, remove
from os.path import getctime, exists

MODEL_SAVE_PATH = ".cache/models/"

def index(request):
    context = { "train" : {"active_class": "active"} }
    return render(request, 'train/index.html', context)

def handle_model_reload(request):
    selected_model = request.session.get('selected_model')
    models = get_models_info(selected_model)
    return JsonResponse(models, safe=False)

def get_models_info(selected_model = ''):
    files = []
    for _, _, filenames in walk(MODEL_SAVE_PATH):
        for f in filenames:
            last_modified = datetime.fromtimestamp(getctime(MODEL_SAVE_PATH+f)).strftime('%Y-%d-%m')
            files.append({
                'name': f, 
                'modified': last_modified,
                'selected': f == selected_model
            })
    return files

def handle_uploaded_file(request):
    if request.method == 'POST':
        if not request.FILES:
            # should be prevented by JS, but reload page as double-check
            context = { "train" : {"active_class": "active"} }
            return redirect('/train', context)
        filename = request.FILES['filechooser'].name
        if exists(MODEL_SAVE_PATH+filename):
            return HttpResponse(400)
        else:
            store_uploaded_file(request.FILES['filechooser'])
        
        # reload model table after file upload
        context = { "train" : {"active_class": "active"} }
        return redirect('/train', context)
    return HttpResponse(400)

def store_uploaded_file(file):
    with open(MODEL_SAVE_PATH+file.name, 'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)

def handle_delete_model(request):
    if request.method == 'GET':
        filename = request.GET.get('filename', '')
        delete_model(filename)
    return HttpResponse()

def delete_model(filename):
    if exists(MODEL_SAVE_PATH+filename):
        remove(MODEL_SAVE_PATH+filename)

def handle_select_model(request):
    if request.method == 'GET':
        filename = request.GET.get('filename', '')
        request.session['selected_model'] = filename
    return HttpResponse()
        