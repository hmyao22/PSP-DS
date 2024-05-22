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
from models.DCP.DCP import OneModel as DCP_OneModel
import time
from draw_defect import draw_detect_region
from thop import profile
from thop import clever_format
from models.HPA.HPA import OneModel

def test(test_split=0, model_name='SG_one'):
    device = 'cuda:0'
    shot = 5
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
        model = OneModel(cls_type='Novel')
    if model_name =='DSAVL':
        model = DSVP(backbone='vgg16')
    if model_name == 'DCP':
        model = DCP_OneModel(cls_type='Base', split=test_split)

    # with torch.no_grad():
    #     memory_before = torch.cuda.memory_allocated()
    #     output = model(image_tensor)
    #     # Get final memory usage
    #     memory_after = torch.cuda.memory_allocated()
    #     memory_usage = (memory_after - memory_before) / (1024 ** 2)  # in MB
    #     print("Memory usage: ", memory_usage, " MB")

    model = model.to(device)
    save_name = model_name + str(test_split)+'.pth'
    model.load_state_dict(torch.load(os.path.join('weights',save_name), map_location='cuda:0'))
    model.eval()

    """"data set """
    data = our_data.Fs_Data(file_dir=r'D:\IMSN-YHM\FS-defect\FS_dataset', test_class=test_split, shot=shot, mode='test')
    # data = our_data.Neu_Data(file_dir=r'D:\IMSN-YHM\dataset\MSD-Seg2', test_class=test_split, mode='test')
    #data = our_data.FCDD_Data(file_dir=r'D:\IMSN-YHM\dataset\FSSD-12', test_class=test_split, shot=shot, mode='test')
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)


    for index, item in enumerate(tqdm(dataloader, ncols=80)):

        que_img, que_mask, supp_img, supp_mask = item
        que_img = que_img.to(device)
        cat_values = 0
        pred_sum = 0
        delta=0

        for i in range(shot):
            pos_img = supp_img[i].to(device)
            pos_mask = supp_mask[i].to(device)
            t1 = time.time()

            if model_name == 'SG_one':
                logits = model(que_img, pos_img, pos_mask)
                pred = model.get_pred(logits, que_img)[0]
            if model_name == 'HSN':
                pred = model.predict_mask(que_img, pos_img, pos_mask)[0]
            if model_name == 'PFENet':
                pos_img = pos_img.unsqueeze(1)
                que_mask = que_mask.squeeze(1)
                pred = model(que_img, pos_img, pos_mask, que_mask)[0]
            if model_name == 'PMM':
                logits = model(que_img, pos_img, pos_mask)
                pred = model.get_pred(logits, que_img)[0]
            if model_name == 'SSP':
                out_ls = model(pos_img, pos_mask, que_img, que_mask.cuda())
                pred = torch.argmax(out_ls[0], dim=1)[0]
            if model_name == 'VAT':
                pred = torch.softmax(model(que_img, pos_img, pos_mask.squeeze(1)),1)[0]
                # pred = pred.argmax(dim=1)
                # pred[pred < 0.5] = 0
                # pred[pred >= 0.5] = 1
            if model_name == 'DCP':
                pos_img = pos_img.unsqueeze(0)
                que_mask = que_mask.squeeze(0)
                pred = model(que_img, pos_img, pos_mask, que_mask)[0]
            if model_name == 'HPA':
                pos_img = pos_img.unsqueeze(0)
                que_mask = que_mask.squeeze(0)
                pred = model(que_img, pos_img, pos_mask, que_mask)[0]
            if model_name == 'DSAVL':
                result = model.get_pred(que_img, pos_img, pos_mask)
                pred_result, forground_similarity_map, background_similarity_map, \
                forground_probability, background_probability = result

                pred = torch.cat([result[-2], result[-1]], dim=0)



            """"running time"""
            t2 = time.time()
            delta += t2 - t1
            pred_sum += pred

        print(delta)
        tmp_pred = pred_sum.cpu().data.numpy()


        if model_name == 'HSN' or model_name =='SSP':
            seg_pred = tmp_pred
            plt.subplot(121)
            plt.imshow(our_data.denormalize(que_img[0].cpu().data.numpy()))
            plt.xticks([])
            plt.yticks([])
            plt.subplot(122)
            plt.imshow(seg_pred, cmap='binary_r')
            plt.xticks([])
            plt.yticks([])
            plt.savefig('result/' + str(index) + '.png')


        elif model_name =='DSAVL':
            plt.subplot(321)
            plt.imshow(our_data.denormalize(que_img[0].cpu().data.numpy()))
            plt.subplot(322)
            plt.imshow(pred_result.cpu().data.numpy())
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.subplot(323)
            plt.imshow(forground_similarity_map.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='inferno')
            plt.subplot(324)
            plt.imshow(background_similarity_map.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='inferno')
            plt.subplot(325)
            plt.imshow(forground_probability.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='inferno')
            plt.colorbar()
            plt.subplot(326)
            plt.imshow(background_probability.squeeze(0).squeeze(0).cpu().data.numpy(), cmap='inferno')
            plt.colorbar()
            plt.savefig('result/' + str(index) + '.png')

        else:
            seg_pred = Seg_head(tmp_pred)

            plt.subplot(141)
            plt.imshow(our_data.denormalize(que_img[0].cpu().data.numpy()))
            plt.xticks([])
            plt.yticks([])
            plt.subplot(142)
            plt.imshow(tmp_pred[0], cmap='binary_r')
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.subplot(143)
            plt.imshow(tmp_pred[1], cmap='binary_r')
            plt.xticks([])
            plt.yticks([])
            plt.colorbar()
            plt.subplot(144)
            plt.imshow(seg_pred, cmap='binary_r')
            plt.xticks([])
            plt.yticks([])
            plt.savefig('result/' + str(index) + '.png')


        # result = draw_detect_region(our_data.denormalize(que_img[0].cpu().data.numpy()),
        #                             pred_result_sum.cpu().data.numpy()*255)
        # gt = draw_detect_region(our_data.denormalize(que_img[0].cpu().data.numpy()),
        #                         que_mask.squeeze().cpu().data.numpy()*255)
        # plt.figure(dpi=400)
        # plt.subplot(131)
        # plt.imshow(our_data.denormalize(que_img[0].cpu().data.numpy()))
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplot(132)
        # plt.imshow(result)
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplot(133)
        # plt.imshow(gt)
        # plt.xticks([])
        # plt.yticks([])
        # plt.tight_layout()
        #

        plt.close()



if __name__ =="__main__":
    test(test_split=0, model_name='DSAVL')
    # test(test_split=1, model_name='DCP')
    # test(test_split=2, model_name='DCP')