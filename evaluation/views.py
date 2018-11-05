from django.shortcuts import render

# Create your views here.
def index(request):
    context = {
        "evaluation" : {"active_class": "active"}
    }
    return render(request, 'evaluation/index.html', context)