from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    path('', views.overview, name='index'),
    path('index.html', views.overview),
    path('overview.html', views.overview),
    path('models.html', views.models),
    path('training.html', views.training),
    path('tensorboard.html', views.tensorboard),

    url('reloadmodel', views.handle_model_reload),
    url('deletemodel', views.handle_delete_model),
    url('selectmodel', views.handle_select_model),
    url('uploadfile', views.handle_uploaded_file),

    url('start_training', views.handle_start_training),
    url('proc_info', views.handle_proc_info),
    url('proc_delete', views.handle_proc_delete)
]