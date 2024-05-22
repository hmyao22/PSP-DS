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
from models.PMM.networks import FPMMs_vgg
from models.OUR_Model import DSVP
from models.SSP.SSP_matching import SSP_MatchingNet
import matplotlib

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


def test():
    device = 'cuda:0'
    model = DSVP(backbone='vgg16')
    shot = 5
    model = model.to(device)
    model.load_state_dict(torch.load('weights/ours/model0.pth', map_location='cuda:0'))
    model.eval()

    image1, label1 = our_data.load_image(r'D:\IMSN-YHM\FS-defect\FS_dataset\class_3\img_4.png',
                                 r'D:\IMSN-YHM\FS-defect\FS_dataset\class_3\label_4.png')
    image1, label1 = our_data.train_transform(image1, label1)

    image2, label2 = our_data.load_image(r'D:\IMSN-YHM\FS-defect\FS_dataset\class_2\img_12.png',
                                 r'D:\IMSN-YHM\FS-defect\FS_dataset\class_2\label_12.png')
    image2, label2 = our_data.train_transform(image2, label2)


    image3, label3 = our_data.load_image(r'D:\IMSN-YHM\FS-defect\FS_dataset\class_5\img_7.png',
                                 r'D:\IMSN-YHM\FS-defect\FS_dataset\class_5\label_7.png')
    image3, label3 = our_data.train_transform(image3, label3)


    image4, label4 = our_data.load_image(r'D:\IMSN-YHM\FS-defect\FS_dataset\class_6\img_7.png',
                                 r'D:\IMSN-YHM\FS-defect\FS_dataset\class_6\label_7.png')
    image4, label4 = our_data.train_transform(image4, label4)

    image5, label5 = our_data.load_image(r'D:\IMSN-YHM\FS-defect\FS_dataset\class_1\img_7.png',
                                         r'D:\IMSN-YHM\FS-defect\FS_dataset\class_1\label_7.png')
    image5, label5 = our_data.train_transform(image5, label5)

    image6, label6 = our_data.load_image(r'D:\IMSN-YHM\FS-defect\FS_dataset\class_7\img_7.png',
                                         r'D:\IMSN-YHM\FS-defect\FS_dataset\class_7\label_7.png')
    image6, label6 = our_data.train_transform(image6, label6)




    test_image,test_label = our_data.load_image(r'D:\IMSN-YHM\FS-defect\2-1.png',
                                 r'D:\IMSN-YHM\FS-defect\FS_dataset\class_5\label_22.png')

    test_image, test_label = our_data.test_transform(test_image, test_label)



    result = model.get_pred(test_image.unsqueeze(0).cuda(), image1.unsqueeze(0).cuda(),
                            label1.unsqueeze(0).unsqueeze(1).cuda())
    _, _, _, forground_probability1, background_probability1 = result


    result = model.get_pred(test_image.unsqueeze(0).cuda(), image2.unsqueeze(0).cuda(),
                            label2.unsqueeze(0).unsqueeze(1).cuda())
    _, _, _, forground_probability2, background_probability2 = result

    result = model.get_pred(test_image.unsqueeze(0).cuda(), image3.unsqueeze(0).cuda(),
                            label3.unsqueeze(0).unsqueeze(1).cuda())
    _, _, _, forground_probability3, background_probability3 = result

    result = model.get_pred(test_image.unsqueeze(0).cuda(), image4.unsqueeze(0).cuda(),
                            label4.unsqueeze(0).unsqueeze(1).cuda())
    _, _, _, forground_probability4, background_probability4 = result

    result = model.get_pred(test_image.unsqueeze(0).cuda(), image5.unsqueeze(0).cuda(),
                            label5.unsqueeze(0).unsqueeze(1).cuda())
    _, _, _, forground_probability5, background_probability5 = result

    result = model.get_pred(test_image.unsqueeze(0).cuda(), image6.unsqueeze(0).cuda(),
                            label6.unsqueeze(0).unsqueeze(1).cuda())
    _, _, _, forground_probability6, background_probability5 = result



    # forground_probability = forground_probability1+forground_probability2\
    #                         +forground_probability3+forground_probability4\
    #                         +forground_probability5
    # background_probability = background_probability1+background_probability2\
    #                          +background_probability3+background_probability4\
    #                          +forground_probability5

    forground_probability = forground_probability5
    background_probability = background_probability5

    # forground_probability = (forground_probability - forground_probability.min()) / (
    #         forground_probability.max() - forground_probability.min())
    # background_probability = (background_probability - background_probability.min()) / (
    #         background_probability.max() - background_probability.min())

    output = torch.cat([forground_probability, background_probability], dim=0)
    print(output.shape)
    final_output = F.upsample(output.unsqueeze(0), size=(256, 256), mode='bilinear')
    _, pred = torch.min(final_output, dim=1)
    pred = pred.squeeze()


    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.)

    plt.subplot(221)
    plt.imshow(our_data.denormalize(test_image.cpu().data.numpy()))
    plt.subplot(222)
    plt.imshow(pred.cpu().data.numpy())
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(223)
    plt.imshow(forground_probability.squeeze(0).squeeze(0).cpu().data.numpy(),cmap='inferno', norm=norm)
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(background_probability.squeeze(0).squeeze(0).cpu().data.numpy(),cmap='inferno', norm=norm)
    plt.colorbar()
    plt.show()



if __name__ == '__main__':
    test()
