from project_conf import MODEL_SAVE_PATH, PROCESS_DIR, TRAINING_PREFIX, CACHE_DIR

import os
from django.shortcuts import redirect
from django.http import HttpResponse

from preprocess.crawler import test_connection
from project_conf import API_KEY_LOCATION

def handle_save_api_key(request):
	if request.method == 'POST':
		api_key = request.POST['api_key']
		status = test_connection(api_key)
		if status == 1:
			save_api_key(api_key)
			return HttpResponse()
		elif status == 0:
			return HttpResponse(status=409)
		elif status == -1:
			return HttpResponse(status=500)
	return HttpResponse(status=405)

def save_api_key(api_key):
	#if os.path.exists(API_KEY_LOCATION):
	#	os.remove(API_KEY_LOCATION)
	with open(API_KEY_LOCATION, 'w') as f:
		f.write(api_key)