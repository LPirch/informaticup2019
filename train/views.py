import os
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django import forms
from time import strftime, ctime
from datetime import datetime
from keras.models import load_model
import keras.backend as K

MODEL_SAVE_PATH = os.path.join('data', 'models')

def overview(request):
    selected_model = request.session.get('selected_model')
    context = { 
        "train" : {"active_class": "active"},
        "selected_model": "None"
    }
    if selected_model:
        model_info = get_model_summary(selected_model)
        context['layers'] = model_info
        context['selected_model'] = selected_model
    
    return render(request, 'train/overview.html', context)

def models(request):
    selected_model = request.session.get('selected_model')
    context = { 
        "train" : {"active_class": "active"},
        "selected_model": "None"
    }
    
    return render(request, 'train/models.html')

def training(request):
    return render(request, 'train/training.html')

def tensorboard(request):    
    return render(request, 'train/tensorboard.html')

def handle_model_reload(request):
    selected_model = request.session.get('selected_model')
    models = get_models_info(selected_model)
    return JsonResponse(models, safe=False)

def get_models_info(selected_model = ''):
    files = []
    for _, _, filenames in os.walk(MODEL_SAVE_PATH):
        for f in filenames:
            last_modified = datetime.fromtimestamp(os.path.getctime(MODEL_SAVE_PATH+f)).strftime('%Y-%d-%m')
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
            return redirect('/train/models.html', context)
        filename = request.FILES['filechooser'].name
        if os.path.exists(os.path.join(MODEL_SAVE_PATH, filename)):
            return HttpResponse(400)
        else:
            store_uploaded_file(request.FILES['filechooser'])
        
        # reload model table after file upload
        context = { "train" : {"active_class": "active"} }
        return redirect('/train/models.html', context)
    return HttpResponse(400)

def store_uploaded_file(file):
    # create cache directory if not exists
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    
    # save file (chunk-wise for handling large files as well)
    with open(os.path.join(MODEL_SAVE_PATH, file.name), 'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)

def handle_delete_model(request):
    if request.method == 'GET':
        filename = request.GET.get('filename', '')
        delete_model(filename)
        if filename == request.session.get('selected_model'):
            request.session['selected_model'] = None
    return HttpResponse()

def delete_model(filename):
    filepath = os.path.join(MODEL_SAVE_PATH, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

def handle_select_model(request):
    if request.method == 'GET':
        filename = request.GET.get('filename', '')
        request.session['selected_model'] = filename
    return HttpResponse()

def get_model_summary(modelname):
    model = load_model(os.path.join('model', 'trained', modelname))
    K.clear_session()
    layer_info = []
    for layer in model.layers:
        layer_info.append({
            'name': layer.name,
            'input_shape': layer.input_shape,
            'output_shape': layer.output_shape
        })
    
    return layer_info
