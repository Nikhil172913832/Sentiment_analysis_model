from django.urls import path
from . import views
urlpatterns = [
    path("", views.index, name="index"),
    path("add/", views.add_person, name = "add person"),
    path("get/", views.get_all_person, name = "get all person"),
]