from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    path('', views.overview, name='index'),
    path('index.html', views.overview),
    path('overview.html', views.overview),
    path('details.html', views.details),
    url('attack.html', views.attack),

    url('proc_info', views.handle_proc_info),
    url('start_attack', views.handle_start_attack),
]