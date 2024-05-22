from models import SG_one
from util import our_data
from torch.utils.data import DataLoader
import torch.optim as opti
from tqdm import tqdm
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
import shutil
import argparse
import models.hsnet
from models.PFENet import PFENet
from  models.PMM.networks import FPMMs_vgg
from models.OUR_Model import DSVP
from models.SSP.SSP_matching import SSP_MatchingNet
from torch.nn import CrossEntropyLoss
from DSAVL_Five_shot_test import validation
from models.VAT.vat import VAT
from models.HPA.HPA import OneModel
from models.DCP.DCP import OneModel as DCP_OneModel
from gpu_mem_track import MemTracker
gpu_tracker = MemTracker()


def train(train_split=1,model_name='HPA'):
    device ='cuda:0'
    gpu_tracker.track()
    if model_name=='SG_one':
        model = SG_one.OneModel()
    if model_name=='HSN':
        model = models.hsnet.HypercorrSqueezeNetwork(backbone='resnet50', use_original_imgsize=False)
    if model_name=='PFENet':
        model = PFENet.PFENet()
    if model_name=='PMM':
        model = FPMMs_vgg.OneModel()
    if model_name == 'SSP':
        model = SSP_MatchingNet('resnet50', False)
    if model_name == 'VAT':
        model = VAT(use_original_imgsize=False)
    if model_name =='HPA':
        model = OneModel(cls_type='Base', split=train_split)
    if model_name == 'DCP':
        model = DCP_OneModel(cls_type='Base', split=train_split)
    if model_name =='DSAVL':
        model = DSVP(backbone='vgg16')
    save_name = model_name + str(train_split)+'.pth'
    #############################
    model.train()
    model = model.to(device)
    gpu_tracker.track()
    best_IOU=0
    data = our_data.Fs_Data(file_dir=r'D:\IMSN-YHM\FS-defect\FS_dataset', test_class=train_split, shot=1, mode='train', HPA=True)
    dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    if model_name =='HPA' or model_name =='DCP':
        optimizer = model.get_optim(model, LR=0.005)
    else:
        optimizer = opti.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-4)

    for epoch in range(100):
        running_loss = 0.0

        for index, item in enumerate(tqdm(dataloader, ncols=80)):
            image, label, support_image_list, support_label_list, query_index = item
            image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)

            for support_image, support_label in zip(support_image_list, support_label_list):
                support_image = support_image.to(device, non_blocking=True)
                support_label = support_label.to(device, non_blocking=True)

            if model_name == 'SG_one':
                logits = model(image, support_image, support_label)
                loss_val = model.get_loss(logits, label)
            if model_name == 'HSN':
                output = model(image, support_image, support_label)
                loss_val = model.compute_objective(output, label)
            if model_name == 'PFENet':
                support_image = support_image.unsqueeze(1)
                label = label.squeeze(1)
                logits = model(image, support_image, support_label, label)
                loss_val = logits[1]+logits[2]
            if model_name == 'PMM':
                logits = model(image, support_image, support_label)
                loss_val = model.get_loss(logits, label)
            if model_name == 'SSP':
                loss_val = model.loss(support_image, support_label, image, label)
            if model_name == 'VAT':
                output = model(image, support_image, support_label.squeeze(1))
                loss_val = model.compute_objective(output, label)
            if model_name == 'HPA':
                support_image = support_image.unsqueeze(0)
                label = label.squeeze(0)
                output = model(image, support_image, support_label, label, [query_index.to(device)])
                loss_val = output[1] + output[2]
            if model_name == 'DCP':
                support_image = support_image.unsqueeze(0)
                label = label.squeeze(0)
                output = model(image, support_image, support_label, label, [query_index.to(device)])
                loss_val = output[1] + output[2]
            if model_name == 'DSAVL':
                D_kl_loss, loss_bce_seg = model.loss(image, label, support_image, support_label)
                loss_val = D_kl_loss+1*loss_bce_seg


            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            running_loss += loss_val.item()

            if index == len(dataloader) - 1:
                print(f"[{epoch}]  F_loss: {(running_loss / (1 * len(dataloader))):.3f}")
                model_dict = model.state_dict()
                torch.save(model_dict, os.path.join('weights',save_name))

        # if model_name =="DSAVL":
        #     IOU = validation(model, index=train_split)
        #     if IOU > best_IOU:
        #         best_IOU = IOU
        #         model_dict = model.state_dict()
        #         torch.save(model_dict, os.path.join('weights', save_name))








if __name__=="__main__":

    for split in [1]:
        for model_name in ['DSAVL']:
            train(train_split=split, model_name=model_name)



