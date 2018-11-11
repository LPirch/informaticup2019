from django.shortcuts import render

# Create your views here.
def index(request):
    context = {
        "attack" : {"active_class": "active"}
    }
    return render(request, 'attack/index.html', context)