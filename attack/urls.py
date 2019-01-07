from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
	path('', views.overview, name='index'),
	path('index.html', views.overview),
	path('overview.html', views.overview),
	path('details.html', views.details),
	url('attack.html', views.attack),

	# GET
	url('proc_info', views.handle_proc_info),
	url('list_images', views.handle_list_images),

	# POST
	url('start_attack', views.handle_start_attack),
	url('delete_proc', views.handle_delete_proc)
]