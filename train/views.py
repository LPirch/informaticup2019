from django.shortcuts import render

# Create your views here.
def index(request):
    context = {
        "train" : {"active_class": "active"}
    }
    return render(request, 'train/index.html', context)