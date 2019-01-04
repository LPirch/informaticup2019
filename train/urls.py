from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
	path('', views.overview, name='index'),
	path('index.html', views.overview),
	path('overview.html', views.overview),
	path('training.html', views.training),
	path('details.html', views.details),

	url('deletemodel', views.handle_delete_model),
	url('uploadmodel', views.handle_upload_model),
	url('getselected', views.handle_get_selected),

	url('start_training', views.handle_start_training),
	url('abort_training', views.handle_abort_training),
	#url('proc_info', views.handle_proc_info),
	#url('proc_delete', views.handle_proc_delete)

	url('start_model_info', views.handle_start_model_info),
	url('model_info', views.handle_model_info)
	
]