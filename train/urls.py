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

    url('models/reloadmodel', views.handle_model_reload),
    url('models/deletemodel', views.handle_delete_model),
    url('models/selectmodel', views.handle_select_model),
    url('models/uploadfile', views.handle_uploaded_file)
]