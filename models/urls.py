from django.urls import path
from django.conf.urls import url

from . import views
from . import rest

urlpatterns = [
	path('', views.overview, name='index'),
	path('index.html', views.overview),
	path('overview.html', views.overview),
	path('training.html', views.training),
	path('details.html', views.details),

	# GET
	url('model_info', rest.handle_model_info),

	# POST
	url('deletemodel', rest.handle_delete_model),
	url('uploadmodel', rest.handle_upload_model),

	url('start_training', rest.handle_start_training),
	url('abort_training', rest.handle_abort_training),
]