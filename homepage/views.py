from typing import NewType
from django.http.response import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


import torch
from torch.nn.functional import threshold
import torchvision
# from .forms import UploadFileForm
from django.core.files.storage import FileSystemStorage

import shutil

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from PIL import Image

# Create your views here.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "./homepage/static/MLmodels/faceDetection.pth"
cfg.DATASETS.TEST = ()
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85   # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.DEVICE='cpu'
predictor = DefaultPredictor(cfg)

new_net = torchvision.models.resnet18()

def getModel(new_net):
    new_net.fc = torch.nn.Linear(512, 4)
    new_net.load_state_dict(torch.load('./homepage/static/MLmodels/classification.pth', map_location="cpu"))

getModel(new_net)

def predict(src):
    result = []
    resultImg = []
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((30,30)),
        torchvision.transforms.ToTensor(),
    ])
    img = cv2.imread(src)
    width,height, _ = img.shape
    sqrtarea = np.sqrt(width * height)
    outputs = predictor(img)  
    for i, box in enumerate(outputs["instances"].pred_boxes):
        x1, y1, x2, y2 = map(int, box.tolist())
        newimg = img[y1:y2,x1:x2]
        color_coverted = cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)
        pil_image=Image.fromarray(color_coverted)   
        tensorimg = trans(pil_image)
        with torch.no_grad():
            new_net.eval()
            prediction_result = new_net(tensorimg.unsqueeze(0))
            result.append(prediction_result.tolist()[0])
            resultImg.append([x1,y1,x2,y2])

    maxindex = np.argmax(np.array(result), 0)
    name = ['엄마','아빠','나','누나']

    li = []
    threshold = 0.8
    for i in range(4):
        dic = {}
        dic["name"] = name[i]
        print(result[maxindex[i]][i])
        path = "/static/image/result/"
        if result[maxindex[i]][i] < threshold:
            cv2.imwrite("./homepage{}{}.png".format(path,i),np.ones((30,30,3), np.uint8)*100)
            dic["prob"] = 0
        else:
            x1,y1,x2,y2 = resultImg[maxindex[i]]
            cv2.imwrite("./homepage{}{}.png".format(path,i), img[y1:y2,x1:x2])
            dic["prob"] = str(round(np.exp(result[maxindex[i]][i]) / (1+np.exp(result[maxindex[i]][i]))*100,1)) + "%"
            cv2.rectangle(img, (x1,y1), (x2,y2), (200,80,80), int(sqrtarea/180))
        dic["img"] = "{}{}.png".format(path,i)
        cv2.imwrite("./homepage" + path + "boxed.png", img)
        li.append(dic)
    return {'result':li, 'img':path + "boxed.png"}

def index(request):

    # form = UploadFileForm()
    return render(request, 'index.html')


def ajax(request):
    dict = {}
    image = request.FILES.get('getImage','')

    if image:
        fs = FileSystemStorage()
        shutil.rmtree('./media/')
        filename  = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)

        dict = predict('./media/' + image.name)

    return HttpResponse(json.dumps(dict), content_type = "application/json")
    

