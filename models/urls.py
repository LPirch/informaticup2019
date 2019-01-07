from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
	path('', views.overview, name='index'),
	path('index.html', views.overview),
	path('overview.html', views.overview),
	path('training.html', views.training),
	path('details.html', views.details),

	# GET
	url('model_info', views.handle_model_info),

	# POST
	url('deletemodel', views.handle_delete_model),
	url('uploadmodel', views.handle_upload_model),

	url('start_training', views.handle_start_training),
	url('abort_training', views.handle_abort_training),
]