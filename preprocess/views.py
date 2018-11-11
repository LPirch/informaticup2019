from django.shortcuts import render

# Create your views here.
def index(request):
    context = {
        "preprocess" : {"active_class": "active"}
    }
    return render(request, 'preprocess/index.html', context)