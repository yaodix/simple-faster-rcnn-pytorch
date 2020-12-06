
import fire
import torch
import torchvision
import numpy as np
# m = torchvision.models.detection.maskrcnn_resnet50_fpn(True)
mm =torchvision.models.detection.fasterrcnn_resnet50_fpn(False)
a= np.array([[1,2],
             [3,4],
             [5,6]
             ])

argmax_ious = np.argmax(a,axis=1)
print(argmax_ious)

max_ious = a[np.arange(a.shape[0]), argmax_ious]
print(max_ious)

gt_argmax_ious = np.argmax(a,axis=0)
print(gt_argmax_ious)
gt_max_ious = a[gt_argmax_ious, np.arange(a.shape[1])]

gt_argmax_ious = np.where(a == gt_max_ious)[0]
print(gt_argmax_ious)

pass