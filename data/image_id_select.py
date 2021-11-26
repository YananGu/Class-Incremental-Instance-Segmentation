import pycocotools
from pycocotools import mask as maskUtils
import torch
import torch.utils.data as data
from pycocotools.coco import COCO

def image_id_select(pre_classes):
    coco = COCO('/data2/gyn/PycharmProjects/AAAI2021ATTMPT/detectron2/tools/datasets/coco/annotations/instances_train2014.json')
    res_bbox = coco.loadRes('results/bbox_detections.json')
    #res_mask = coco.loadRes('results/mask_detections.json')
    annos = res_bbox.anns
    list_total=[]

    for i in pre_classes:
        list_=[]
        for key,value in annos.items():
            if value['category_id'] == i  and value['score'] > 0.5 :
                list_.append(value['image_id'])
                list_ = list(set(list_))
    list_total.append(list_)

    return list_total




