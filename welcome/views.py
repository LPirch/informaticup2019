from project_conf import API_KEY_LOCATION

import os
from django.shortcuts import render

def welcome(request):
	context = {}
	if not os.path.exists(API_KEY_LOCATION):
		context['apikey_missing'] = True
	return render(request, 'welcome.html', context)