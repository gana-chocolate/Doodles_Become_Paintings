from django.shortcuts import render,reverse
from django.http import HttpResponseRedirect
from django import forms
from .forms import UploadFileForm
from .forms import UploadFileForm2
from .forms import UploadFileModel
# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from mainapp.models import UploadFileModel
from mainapp.models import Post
import os
from django.conf import settings
import os
import base64
from PIL import Image
import PIL.ImageOps
    from .forms import FileFieldForm
from django.views.generic.edit import FormView
import shutil
import test
import glob, os, os.path

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect(reverse('/mainapp/'))
    else:
        '''form = UploadFileForm()'''
    print(form.files)
    for filename, file in request.FILES.items():
        name = request.FILES[filename].name
    print(name)
    post = UploadFileForm2(data=request.POST, files=request.FILES)
    post = Post()
    post.profile_pic = request.FILES.get('uploadfile')
    post.pdf = request.FILES.get('uploadfile')
    print(request.FILES.get('uploadfile'))
    post.save()

    '''
    uploadfilemodel = UploadFileModel()
    uploadfilemodel.title = request.POST.get('uploadfile', None)
    print(request.POST.get('uploadfile'))
    uploadfilemodel.save()'''
    return render(request, 'mainapp/index.html')


class FileFieldView(FormView):
    form_class = FileFieldForm
    template_name = 'upload.html'  # Replace with your template.
    success_url = '...'  # Replace with your URL or reverse().
    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('file_field')
        if form.is_valid():
            for f in files:
                ...  # Do something with each file.
            return self.form_valid(form)
        else:
            return self.form_invalid(form)

@csrf_exempt
def send(request):
    tesffff = request.POST['imgBase64']
    img_b64 = base64.b64decode(tesffff[22:])
    f1 = open('mainapp/test.jpg', 'wb')
    f1.write(img_b64)
    im = Image.open("mainapp/test.jpg")
    bg = Image.new("RGB",im.size,(255,255,255))
    bg.paste(im,(0,0),im)
    bg.save('mainapp/media/test.jpg')
    return render(request, 'mainapp/index.html')

@csrf_exempt
def send2(request):
    tesffff2 = request.POST['imgBase64']
    print(tesffff2[22:])
    print("씨발")
    img_b642 = base64.b64decode(tesffff2[22:])
    f2 = open('mainapp/media/test.jpg', 'wb')
    f2.write(img_b642)
    #test()
    im2 = Image.open("mainapp/media/test.png")
    bg2 = Image.new("RGB",im2.size,(255,255,255))
    bg2.paste(im2,(0,0),im2)
    bg2.save('mainapp/media/test.jpg')
    #f2 = open('/Users/janghyukjin/WorkSpace/prison.png', 'wb')
    #f2.write(img_b64)
    return render(request, 'mainapp/index.html')


def index(request):
    template = loader.get_template('mainapp/index.html')
    context = {
        'latest_question_list': "test",
    }
    return HttpResponse(template.render(context, request))


def download(request):
    print("####")
    file_ = open(os.path.join(settings.MEDIA_ROOT, 'prison.jpg'))
    path = settings.MEDIA_ROOT
    file_list = os.listdir(path)
    print("file_list: {}".format(file_list))
    return render(request, 'mainapp/index.html')
