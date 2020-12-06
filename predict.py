import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import matplotlib.pyplot as plt
import cv2
img = read_image("C:\\MyData\\ped.jpg")
img = t.from_numpy(img)[None]
show_img = cv2.imread("C:\\MyData\\ped.jpg")

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

trainer.load(R'C:\detection\simple-faster-rcnn-pytorch\model\fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth.701052458187')
opt.caffe_pretrain=False # this model was trained from torchvision-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img,visualize=True)
ax = vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))
# it failed to find the dog, but if you set threshold from 0.7 to 0.6, you'll find it
boxes = at.tonumpy(_bboxes[0])
for b in boxes:
    cv2.rectangle(show_img, (int(b[1]),int(b[0])) ,(int(b[3]),int(b[2])),(0,0,255),1)

cv2.imwrite("c:\\tes.jpg",show_img)
