from django.urls import path
from django.conf.urls import url

from . import views
from . import rest

urlpatterns = [
	path('', views.welcome, name='index'),
	path('index.html', views.welcome),
	path('welcome.html', views.welcome),

	# POST
	url('save_api_key', rest.handle_save_api_key),
]