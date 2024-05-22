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
import matplotlib.pyplot as plt
from util import our_data
import models.hsnet
from models.PFENet import PFENet
from  models.PMM.networks import FPMMs_vgg
from models.OUR_Model import DSVP
from models.SSP.SSP_matching import SSP_MatchingNet


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
    IOU = tp/(tp+fp+fn)
    return IOU



def test(test_split=2):
    device = 'cuda:0'
    model = DSVP(backbone='vgg16')
    shot = 1

    model = model.to(device)
    save_name = 'DSAVL' + str(test_split) + '.pth'
    model.load_state_dict(torch.load(os.path.join('weights', save_name), map_location='cuda:0'))
    model.eval()
    data = our_data.Fs_Data(file_dir=r'D:\IMSN-YHM\FS-defect\FS_dataset', test_class=test_split, shot=shot, mode='test')
    #data = our_data.Neu_Data(file_dir=r'D:\IMSN-YHM\dataset\MSD-Seg2', test_class=test_split, mode='test')
    #data = our_data.FCDD_Data(file_dir=r'D:\IMSN-YHM\dataset\FSSD-12', test_class=test_split, shot=shot, mode='test')
    dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)


    for index, item in enumerate(tqdm(dataloader, ncols=80)):
        que_img, que_mask, supp_img, supp_mask = item

        que_img = que_img.cuda()
        cat_values = 0
        pred_sum = 0

        pred_result_sum = 0
        forground_similarity_map_sum= 0
        background_similarity_map_sum = 0
        forground_likelihood_sum = 0
        background_likelihood_sum = 0

        pos_imgs=[]
        pos_masks=[]
        for i in range(shot):
            pos_img = supp_img[i].cuda()
            pos_mask = supp_mask[i].cuda()
            pos_imgs.append(pos_img)
            pos_masks.append(pos_mask)


        result = model.five_shot_forward(que_img, pos_imgs, pos_masks)
        pred_result, forground_similarity_map, background_similarity_map, forground_probability,background_probability = result
        pred_result_sum +=pred_result
        forground_similarity_map_sum += forground_similarity_map
        background_similarity_map_sum += background_similarity_map
        forground_likelihood_sum += forground_probability
        background_likelihood_sum += background_probability
        ####### Ours ########

        ############ours###############

        plt.subplot(321)
        plt.imshow(our_data.denormalize(que_img[0].cpu().data.numpy()))
        plt.subplot(322)
        plt.imshow(pred_result_sum.cpu().data.numpy(), cmap='Greys')


        plt.xticks([])
        plt.yticks([])
        plt.subplot(323)
        plt.imshow(forground_similarity_map_sum.squeeze(0).squeeze(0).cpu().data.numpy(),cmap='inferno')
        plt.colorbar()
        plt.subplot(324)
        plt.imshow(background_similarity_map_sum.squeeze(0).squeeze(0).cpu().data.numpy(),cmap='inferno')
        plt.colorbar()
        plt.subplot(325)
        plt.imshow(forground_likelihood_sum.squeeze(0).squeeze(0).cpu().data.numpy(),cmap='inferno')
        plt.colorbar()
        plt.subplot(326)
        plt.imshow(background_likelihood_sum.squeeze(0).squeeze(0).cpu().data.numpy(),cmap='inferno')
        plt.colorbar()

        plt.savefig('result/'+str(index)+'.png')
        save_result = cv2.applyColorMap(255 * pred_result_sum.cpu().data.numpy().astype(np.uint8), cv2.COLORMAP_OCEAN)
        cv2.imwrite('result/' + 'segmap' + str(index) + '.png', save_result)

        # plt.show()



def measure(test_split=2):
    device ='cuda:0'

    model = DSVP(backbone='vgg16')
    save_name = 'DSAVL' + str(test_split) + '.pth'
    shot = 5
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join('weights', save_name), map_location='cuda:0'))
    model.eval()
    data = our_data.Fs_Data(file_dir=r'D:\IMSN-YHM\FS-defect\FS_dataset', test_class=test_split, shot=shot, mode='test')
    #data = our_data.Neu_Data1(file_dir=r'D:\IMSN-YHM\dataset\MSD-Seg2', test_class=test_split, shot=shot, mode='test')
    #data = our_data.FCDD_Data(file_dir=r'D:\IMSN-YHM\dataset\FSSD-12', test_class=test_split, shot=shot, mode='test')
    dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    number_sample = int(data.__len__())

    result = torch.zeros([number_sample, 256, 256])
    label = torch.zeros([number_sample, 256, 256])


    for index, item in enumerate(tqdm(dataloader, ncols=80)):
        que_img, que_mask, supp_img, supp_mask = item
        que_img = que_img.cuda()


        pos_imgs=[]
        pos_masks=[]
        for i in range(shot):
            pos_img = supp_img[i].cuda()
            pos_mask = supp_mask[i].cuda()
            pos_imgs.append(pos_img)
            pos_masks.append(pos_mask)

        ##########ours #########
        results = model.five_shot_forward(que_img, pos_imgs, pos_masks)
        pred_result, _, _, _, _ = results

        tmp_pred = pred_result.cpu().data
        tmp_pred[tmp_pred > 0.] = 1.

        result[index] = tmp_pred
        label[index] = que_mask

    result = result.flatten()
    label = label.flatten()
    IOU = IOU_measure(result, label)
    print(IOU)
    FB_IOU = (IOU_measure(result, label, is_pos=True) +IOU_measure(result, label, is_pos=False))/2
    print(FB_IOU)
    return IOU,FB_IOU



def validation(model,index=0):
    device ='cuda:0'

    shot = 5
    model = model.to(device)
    model.eval()
    data = our_data.Fs_Data(file_dir=r'D:\IMSN-YHM\FS-defect\FS_dataset', test_class=index, shot=shot, mode='test')
    # data = our_data.Neu_Data(file_dir=r'D:\IMSN-YHM\dataset\MSD-Seg2', test_class=0, mode='test')
    dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    number_sample = int(data.__len__())

    result = torch.zeros([number_sample, 256, 256])
    label = torch.zeros([number_sample, 256, 256])

    for index, item in enumerate(tqdm(dataloader, ncols=80)):
        que_img, que_mask, supp_img, supp_mask = item
        que_img = que_img.cuda()

        pos_imgs=[]
        pos_masks=[]
        for i in range(shot):
            pos_img = supp_img[i].cuda()
            pos_mask = supp_mask[i].cuda()
            pos_imgs.append(pos_img)
            pos_masks.append(pos_mask)

        ##########ours #########
        results = model.five_shot_forward(que_img, pos_imgs, pos_masks)
        pred_result, _, _, _, _ = results

        tmp_pred = pred_result.cpu().data

        result[index] = tmp_pred
        label[index] = que_mask


    result = result.flatten()
    label = label.flatten()
    IOU = IOU_measure(result, label)
    print(IOU)
    FB_IOU = (IOU_measure(result, label, is_pos=True) +IOU_measure(result, label, is_pos=False))/2
    print(FB_IOU)
    model.train()
    return IOU





if __name__ =='__main__':
    test(test_split=0)
    IOU1, FB_IOU1 = measure(test_split=0)
    IOU2, FB_IOU2 = measure(test_split=1)
    IOU3, FB_IOU3 = measure(test_split=2)

    print('mean IOU' + str((IOU1 + IOU2 + IOU3) / 3))
    print('mean FBIOU' + str((FB_IOU1 + FB_IOU2 + FB_IOU3) / 3))
    # test()
