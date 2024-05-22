from models import SG_one
from util import our_data
from torch.utils.data import DataLoader
import torch.optim as opti
from tqdm import tqdm
import cv2
import numpy as np
import os
import torch
from util.util import Seg_head
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
import shutil
import argparse
import matplotlib.pyplot as plt
from util import our_data
import models.hsnet
from models.PFENet import PFENet
from  models.PMM.networks import FPMMs_vgg
from models.OUR_Model import DSVP
from models.SSP.SSP_matching import SSP_MatchingNet
from models.VAT.vat import VAT
import time
from draw_defect import draw_detect_region
from thop import profile
from thop import clever_format
from models.HPA.HPA import OneModel
from models.DCP.DCP import OneModel as DCP_OneModel
from util.util import Seg_head
import time
def IOU_measure(y_in, pred_in, is_pos=True):
    thresh = 0.5
    if is_pos:
        y = y_in > thresh
        pred = pred_in > thresh
    else:
        y = y_in < thresh
        pred = pred_in < thresh
    tp = np.logical_and(y, pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    IOU = tp / (tp + fp + fn)
    return IOU


def measure(test_split=1, model_name='SG_one'):
    device = 'cuda:0'
    shot = 5
    #### model selectin #####
    if model_name == 'SG_one':
        model = SG_one.OneModel()
    if model_name == 'HSN':
        model = models.hsnet.HypercorrSqueezeNetwork(backbone='resnet50', use_original_imgsize=False)
    if model_name == 'PFENet':
        model = PFENet.PFENet(shot=shot)
    if model_name == 'PMM':
        model = FPMMs_vgg.OneModel()
    if model_name == 'SSP':
        model = SSP_MatchingNet('resnet50', False)
    if model_name == 'VAT':
        model = VAT(use_original_imgsize=False)
    if model_name == 'HPA':
        model = OneModel(cls_type='Novel',shot=shot)
    if model_name == 'DCP':
        model = DCP_OneModel(cls_type='Base', split=test_split, shot=shot)
    if model_name == 'DSAVL':
        model = DSVP(backbone='vgg16')


    model = model.to(device)
    save_name = model_name + str(test_split) + '.pth'
    #model.load_state_dict(torch.load(os.path.join('weights', save_name), map_location='cuda:0'))
    model.eval()

    data = our_data.Fs_Data(file_dir=r'D:\IMSN-YHM\FS-defect\FS_dataset', test_class=test_split, shot=shot, mode='test')
    dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    number_sample = int(data.__len__())

    result = torch.zeros([number_sample, 256, 256])
    label = torch.zeros([number_sample, 256, 256])

    for index, item in enumerate(tqdm(dataloader, ncols=80)):
        que_img, que_mask, supp_img, supp_mask = item

        que_img = que_img.cuda()

        pred_result_sum = 0
        pos_imgs = []
        pos_masks = []
        for i in range(shot):
            pos_img = supp_img[i].numpy()
            pos_mask = supp_mask[i].numpy()
            pos_imgs.append(pos_img)
            pos_masks.append(pos_mask)
        pos_imgs = np.array(pos_imgs)
        pos_masks = np.array(pos_masks)

        pos_imgs = torch.from_numpy(pos_imgs).cuda()
        pos_masks = torch.from_numpy(pos_masks).cuda()
        t1 =time.time()
        if model_name == 'HPA':
            pos_imgs = pos_imgs.permute(1, 0, 2, 3, 4)
            pos_masks = pos_masks.permute(1, 0, 2, 3, 4).squeeze().unsqueeze(0)
            que_mask = que_mask.squeeze(0)

            pred = model(que_img, pos_imgs, pos_masks, que_mask)[0]
            tmp_pred = pred.cpu().data.numpy()
            seg_pred = Seg_head(tmp_pred)


        if model_name == 'PFENet':
            pos_imgs = pos_imgs.permute(1, 0, 2, 3, 4)
            pos_masks = pos_masks.permute(1, 0, 2, 3, 4).squeeze().unsqueeze(0)
            que_mask = que_mask.squeeze(0)

            pred = model(que_img, pos_imgs, pos_masks, que_mask)[0]
            tmp_pred = pred.cpu().data.numpy()
            seg_pred = Seg_head(tmp_pred)

        if model_name == 'SG_one':
            logits = model.forward_5shot_max(que_img, pos_imgs, pos_masks)
            pred = model.get_pred_5shot_max(logits, que_img)[0]

            tmp_pred = pred.cpu().data.numpy()
            seg_pred = Seg_head(tmp_pred)
        if model_name == 'PMM':
            pos_imgs = pos_imgs.permute(1, 0, 2, 3, 4)
            pos_masks = pos_masks.permute(1, 0, 2, 3, 4)
            logits = model.forward_5shot(que_img, pos_imgs, pos_masks)
            pred = model.get_pred(logits, que_img)[0]
            tmp_pred = pred.cpu().data.numpy()
            seg_pred = Seg_head(tmp_pred)

        if model_name == 'DCP':
            pos_imgs = pos_imgs.permute(1, 0, 2, 3, 4)
            pos_masks = pos_masks.permute(1, 0, 2, 3, 4).squeeze().unsqueeze(0)
            que_mask = que_mask.squeeze(0)

            pred = model(que_img, pos_imgs, pos_masks, que_mask)[0]
            tmp_pred = pred.cpu().data.numpy()
            seg_pred = Seg_head(tmp_pred)
        t2 = time.time()
        print(t2-t1)



        result[index] = torch.from_numpy(seg_pred)
        label[index] = que_mask

    result = result.flatten()
    label = label.flatten()
    IOU = IOU_measure(result, label)
    print(IOU)
    FB_IOU = (IOU_measure(result, label, is_pos=True) + IOU_measure(result, label, is_pos=False)) / 2
    print(FB_IOU)





if __name__ == '__main__':
    measure(test_split=0, model_name='DCP')
    measure(test_split=1, model_name='DCP')
    measure(test_split=2, model_name='DCP')


