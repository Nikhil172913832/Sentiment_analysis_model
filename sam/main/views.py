from django.shortcuts import render
from .models import person_collection
# Create your views here.
from django.http import HttpResponse

def index(request):
    return HttpResponse("Everything is fine!!")
def add_person(request):
    records = {
        "first_name": "John",
        "last_name": "smith"
    }
    person_collection.insert_one(records)
    return HttpResponse("New person is added")
def get_all_person(request):
    person = person_collection.find()
    return HttpResponse(person)