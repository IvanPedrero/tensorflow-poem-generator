from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .models import Greeting
import requests
import json
from .poem_generator import getPoem
import sys
# Create your views here.


def index(request):
    # return HttpResponse('Hello from Python!')
    return render(request, "index.html")


def cute_face(request):
    r = requests.get('http://httpbin.org/status/418')
    print(r.text)
    return HttpResponse('<pre>' + r.text + '</pre>')


def predict_poem(request):

    if request.is_ajax:

        jsonResponse = None

        try:
            # Format the data.
            req_data = request.GET.get('data', None) 
            dataform = str(req_data).strip("'<>() ").replace('\'', '\"')
            data = json.loads(dataform)

            # Get the seed.
            seed = data["seed"]
            seed = seed.strip()

            print("Generating poem...")

            poem = getPoem(seed, False)

            print("Generated poem: " + poem)

            # Send the poem.
            jsonResponse = JsonResponse({"poem": poem}, status=200)
        
        except:
            jsonResponse = JsonResponse({"error": "There was an error processing the sent data."}, status=500)
        
        return jsonResponse

        
