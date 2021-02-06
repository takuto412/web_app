from __future__ import print_function
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import DocumentForm
from .models import Document
import cv2
from django.conf import settings
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
from .super_resolve import super_resolve


def index(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('index')
    else:
        form = DocumentForm()
        max_id = Document.objects.latest('id').id
        obj = Document.objects.get(id = max_id)
        
        input_path = settings.BASE_DIR + obj.photo.url
        output_path = settings.BASE_DIR + "/media/output/output.jpg"
        super_resolve(input_path, output_path)

    return render(request, 'app1/index.html', {
        'form': form,
        'obj':obj,
    })


