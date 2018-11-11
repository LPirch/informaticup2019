from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    url('reloadmodel', views.handle_model_reload),
    url('deletemodel', views.handle_delete_model),
    url('selectmodel', views.handle_select_model),
    url('uploadfile', views.handle_uploaded_file)
]