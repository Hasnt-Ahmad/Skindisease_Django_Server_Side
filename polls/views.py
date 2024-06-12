from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import os
import json
import cv2
from . import model
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
@csrf_exempt
def file_upload_view(request):
    if request.method == 'POST':

        img_file = request.FILES['file']
        img_data = img_file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        predict_im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        try:
            predictions = model.predictDisease(predict_im)
            print("Prediction : ",predictions)
            return JsonResponse({'predictions': predictions})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
