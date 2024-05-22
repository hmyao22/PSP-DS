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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from util import our_data
import models.hsnet
from models.PFENet import PFENet
from  models.PMM.networks import FPMMs_vgg
from models.SSP.SSP_matching import SSP_MatchingNet
from models.OUR_Model import DSVP
from models.VAT.vat import VAT
from models.HPA.HPA import OneModel
from models.DCP.DCP import OneModel as DCP_OneModel
from util.util import Seg_head


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



def measure(test_split=1, model_name='SG_one'):
    device = 'cuda:0'
    shot = 1
    if model_name == 'SG_one':
        model = SG_one.OneModel()
    if model_name == 'HSN':
        model = models.hsnet.HypercorrSqueezeNetwork(backbone='resnet50', use_original_imgsize=False)
    if model_name == 'PFENet':
        model = PFENet.PFENet()
    if model_name == 'PMM':
        model = FPMMs_vgg.OneModel()
    if model_name == 'SSP':
        model = SSP_MatchingNet('resnet50', False)
    if model_name == 'VAT':
        model = VAT(use_original_imgsize=False)
    if model_name == 'HPA':
        model = OneModel(cls_type='Novel')
    if model_name == 'DCP':
        model = DCP_OneModel(cls_type='Base', split=test_split)
    if model_name == 'DSAVL':
        model = DSVP(backbone='vgg16')

    model = model.to(device)
    save_name = model_name + str(test_split) + '.pth'
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
        pred_sum = 0
        for i in range(shot):
            pos_img = supp_img[i].cuda()
            pos_mask = supp_mask[i].cuda()
            # pos_mask[pos_mask > 0.] = 1.

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
                with torch.no_grad():
                    out_ls = model(pos_img, pos_mask, que_img, que_mask.cuda())
                    pred = torch.argmax(out_ls[0], dim=1)
            if model_name == 'VAT':
                pred = torch.softmax(model(que_img, pos_img, pos_mask.squeeze(1)), 1)[0]
            if model_name == 'DCP':
                pos_img = pos_img.unsqueeze(0)
                que_mask = que_mask.squeeze(0)
                pred = model(que_img, pos_img, pos_mask, que_mask)[0]
            if model_name == 'HPA':
                pos_img = pos_img.unsqueeze(0)
                que_mask = que_mask.squeeze(0)
                pred = model(que_img, pos_img, pos_mask, que_mask)[0]
            if model_name == 'DSAVL':
                our_result = model.get_pred(que_img, pos_img, pos_mask)
                pred, _, _, _, _ = our_result
            pred_sum += pred


        tmp_pred = pred_sum.cpu().data.numpy()
        if model_name =='HSN' or model_name =='SSP' or model_name =='DSAVL':
            tmp_pred = tmp_pred
        else:
            tmp_pred = Seg_head(tmp_pred)

        result[index] = torch.from_numpy(tmp_pred)
        label[index] = que_mask

        # plt.subplot(121)
        # plt.imshow(our_data.denormalize(que_img[0].cpu().data.numpy()))
        # plt.subplot(122)
        # plt.imshow(tmp_pred)
        # plt.show()
    result = result.flatten()
    label = label.flatten()
    IOU = IOU_measure(result, label)
    print(IOU)
    FB_IOU = (IOU_measure(result, label, is_pos=True) + IOU_measure(result, label, is_pos=False))/2
    print(FB_IOU)

    # ConfusionMatrixDisplay.from_predictions(label, result, display_labels=["Normal", "Defect"], cmap=plt.cm.Reds,colorbar=True)
    # plt.title("Confusion Matrix")
    # plt.show()
    return IOU, FB_IOU


if __name__ =="__main__":
    IOU1, FB_IOU1 = measure(test_split=0, model_name='DSAVL')
    IOU2, FB_IOU2 = measure(test_split=1, model_name='DSAVL')
    IOU3, FB_IOU3 = measure(test_split=2, model_name='DSAVL')

    print('mean IOU'+str((IOU1+IOU2+IOU3)/3))
    print('mean FBIOU'+str((FB_IOU1+FB_IOU2+FB_IOU3)/3))
