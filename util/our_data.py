import os
import os.path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm
from PIL import Image
import util.transform as transform
from torch.utils.data import DataLoader

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

train_transform = [
    transform.RandRotate([-10, 10], padding=mean, ignore_label=0),
    transform.RandomGaussianBlur(),
    transform.RandomHorizontalFlip(),
    transform.Resize(256),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)]
train_transform = transform.Compose(train_transform)


test_transform = [
    transform.Resize(256),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)]
test_transform = transform.Compose(test_transform)


IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm']

def load_image(image_path, label_path):
    image = cv2.imread(image_path)
    label = cv2.imread(label_path, 0)
    label[label < 50] = 0
    label[label > 50] = 1
    return image, label



def GetFiles(file_dir, file_type, IsCurrent=False):
    file_list = []
    for parent, dirnames, filenames in os.walk(file_dir):
        for filename in filenames:
            for type in file_type:
                if filename.endswith(('.%s' % type)):
                    file_list.append(os.path.join(parent, filename))

        if IsCurrent == True:
            break
    return file_list



# def load_dataset(file_dir):
#     total_data = GetFiles(file_dir, IMG_EXTENSIONS)
#     sub_class_file_list = {}
#     for sub_c in range(7):
#         sub_class_file_list[sub_c] = []
#     for index in tqdm(range(len(total_data))):
#         file_path = total_data[index]
#         file_class = file_path.split('\\')[-2]
#         file_name = file_path.split('\\')[-1]
#         if file_class=='class_1':
#             if file_name.split('_')[0] =="img":
#                label_name = 'label'+'_'+file_name.split('_')[-1]
#                label_path = file_path.replace(file_name, label_name)
#                item = (file_path, label_path)
#                sub_class_file_list[0].append(item)
#         if file_class=='class_2':
#             if file_name.split('_')[0] == "img":
#                 label_name = 'label' + '_' + file_name.split('_')[-1]
#                 label_path = file_path.replace(file_name, label_name)
#                 item = (file_path, label_path)
#                 sub_class_file_list[1].append(item)
#         if file_class=='class_3':
#             if file_name.split('_')[0] == "img":
#                 label_name = 'label' + '_' + file_name.split('_')[-1]
#                 label_path = file_path.replace(file_name, label_name)
#                 item = (file_path, label_path)
#                 sub_class_file_list[2].append(item)
#         if file_class=='class_4':
#             if file_name.split('_')[0] == "img":
#                 label_name = 'label' + '_' + file_name.split('_')[-1]
#                 label_path = file_path.replace(file_name, label_name)
#                 item = (file_path, label_path)
#                 sub_class_file_list[3].append(item)
#         if file_class=='class_5':
#             if file_name.split('_')[0] == "img":
#                 label_name = 'label' + '_' + file_name.split('_')[-1]
#                 label_path = file_path.replace(file_name, label_name)
#                 item = (file_path, label_path)
#                 sub_class_file_list[4].append(item)
#         if file_class=='class_6':
#             if file_name.split('_')[0] == "img":
#                 label_name = 'label' + '_' + file_name.split('_')[-1]
#                 label_path = file_path.replace(file_name, label_name)
#                 item = (file_path, label_path)
#                 sub_class_file_list[5].append(item)
#         if file_class=='class_7':
#             if file_name.split('_')[0] == "img":
#                 label_name = 'label' + '_' + file_name.split('_')[-1]
#                 label_path = file_path.replace(file_name, label_name)
#                 item = (file_path, label_path)
#                 sub_class_file_list[6].append(item)
#     return sub_class_file_list


def load_dataset(file_dir):
    total_data = GetFiles(file_dir, IMG_EXTENSIONS)
    sub_class_file_list = {}
    for sub_c in range(3):
        sub_class_file_list[sub_c] = []
    for index in tqdm(range(len(total_data))):
        file_path = total_data[index]
        file_class = file_path.split('\\')[-2]
        file_name = file_path.split('\\')[-1]
        if file_class=='class_1' or file_class=='class_2' or file_class=='class_3':
            if file_name.split('_')[0] =="img":
               label_name = 'label'+'_'+file_name.split('_')[-1]
               label_path = file_path.replace(file_name, label_name)
               item = (file_path, label_path)
               sub_class_file_list[0].append(item)
        if file_class == 'class_4' or file_class == 'class_5':
            if file_name.split('_')[0] == "img":
                label_name = 'label' + '_' + file_name.split('_')[-1]
                label_path = file_path.replace(file_name, label_name)
                item = (file_path, label_path)
                sub_class_file_list[1].append(item)
        if file_class == 'class_6' or file_class == 'class_7':
            if file_name.split('_')[0] == "img":
                label_name = 'label' + '_' + file_name.split('_')[-1]
                label_path = file_path.replace(file_name, label_name)
                item = (file_path, label_path)
                sub_class_file_list[2].append(item)
    return sub_class_file_list


def load_NEU(file_dir):
    total_data = GetFiles(file_dir, IMG_EXTENSIONS)
    sub_class_file_list = {}
    for sub_c in range(3):
        sub_class_file_list[sub_c] = []
    for index in tqdm(range(len(total_data))):
        file_path = total_data[index]
        file_class = file_path.split('\\')[-3]
        file_type = file_path.split('\\')[-2]
        file_name = file_path.split('\\')[-1]

        if file_class=='AI_Rm' or file_class=='MT_Uneven' or file_class=='Steel_Ld' or file_class=="Steel_Am":
            if file_type =="Images":
               label_name = file_name.split('.')[0]+'.png'
               label_path = file_path.replace(file_name, label_name)
               label_path = label_path.replace(file_type, 'GT')
               item = (file_path, label_path)
               sub_class_file_list[0].append(item)

        if file_class=='leather' or file_class=='Steel_Pa' or file_class=='Rail' or file_class=='Steel_Sc':
            if file_type =="Images":
               label_name = file_name.split('.')[0]+'.png'
               label_path = file_path.replace(file_name, label_name)
               label_path = label_path.replace(file_type, 'GT')
               item = (file_path, label_path)
               sub_class_file_list[1].append(item)
        if file_class=='Al_Con' or file_class=='MT_Fray' or file_class=='MT_Break' or file_class=='tile':
            if file_type =="Images":
               label_name = file_name.split('.')[0]+'.png'
               label_path = file_path.replace(file_name, label_name)
               label_path = label_path.replace(file_type, 'GT')
               item = (file_path, label_path)
               sub_class_file_list[2].append(item)


    return sub_class_file_list


def load_NEU1(file_dir):
    total_data = GetFiles(file_dir, IMG_EXTENSIONS)
    sub_class_file_list = {}
    for sub_c in range(4):
        sub_class_file_list[sub_c] = []
    for index in tqdm(range(len(total_data))):
        file_path = total_data[index]
        file_class = file_path.split('\\')[-3]
        file_type = file_path.split('\\')[-2]
        file_name = file_path.split('\\')[-1]

        if file_class=='AI_Rm' or file_class=='MT_Uneven' or file_class=='Al_Con':
            if file_type =="Images":
               label_name = file_name.split('.')[0]+'.png'
               label_path = file_path.replace(file_name, label_name)
               label_path = label_path.replace(file_type, 'GT')
               item = (file_path, label_path)
               sub_class_file_list[0].append(item)

        if file_class=='MT_Break' or file_class=='MT_Fray' or file_class=='Steel_Ld':
            if file_type =="Images":
               label_name = file_name.split('.')[0]+'.png'
               label_path = file_path.replace(file_name, label_name)
               label_path = label_path.replace(file_type, 'GT')
               item = (file_path, label_path)
               sub_class_file_list[1].append(item)
        if file_class=='Steel_Am' or file_class=='Steel_Pa' or file_class=='Rail':
            if file_type =="Images":
               label_name = file_name.split('.')[0]+'.png'
               label_path = file_path.replace(file_name, label_name)
               label_path = label_path.replace(file_type, 'GT')
               item = (file_path, label_path)
               sub_class_file_list[2].append(item)

        if file_class == 'Steel_Sc' or file_class == 'tile' or file_class == 'tile':
            if file_type == "Images":
                label_name = file_name.split('.')[0] + '.png'
                label_path = file_path.replace(file_name, label_name)
                label_path = label_path.replace(file_type, 'GT')
                item = (file_path, label_path)
                sub_class_file_list[3].append(item)


    return sub_class_file_list


def load_FSSD(file_dir):
    total_data = GetFiles(file_dir, IMG_EXTENSIONS)
    sub_class_file_list = {}
    for sub_c in range(3):
        sub_class_file_list[sub_c] = []
    for index in tqdm(range(len(total_data))):
        file_path = total_data[index]
        file_class = file_path.split('\\')[-3]
        file_type = file_path.split('\\')[-2]
        file_name = file_path.split('\\')[-1]

        if file_class=='Steel_Am' or file_class=='Steel_Ia' or file_class=='Steel_Ld' or file_class=="Steel_Op":
            if file_type =="Images":
               label_name = file_name.split('.')[0]+'.png'
               label_path = file_path.replace(file_name, label_name)
               label_path = label_path.replace(file_type, 'GT')
               item = (file_path, label_path)
               sub_class_file_list[0].append(item)

        if file_class=='Steel_Os' or file_class=='Steel_Ws' or file_class=='Steel_Pa' or file_class=='Steel_Pk':
            if file_type =="Images":
               label_name = file_name.split('.')[0]+'.png'
               label_path = file_path.replace(file_name, label_name)
               label_path = label_path.replace(file_type, 'GT')
               item = (file_path, label_path)
               sub_class_file_list[1].append(item)
        if file_class=='Steel_Ri' or file_class=='Steel_Rp' or file_class=='Steel_Sc' or file_class=='Steel_Se':
            if file_type =="Images":
               label_name = file_name.split('.')[0]+'.png'
               label_path = file_path.replace(file_name, label_name)
               label_path = label_path.replace(file_type, 'GT')
               item = (file_path, label_path)
               sub_class_file_list[2].append(item)


    return sub_class_file_list



def denormalize(img):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


class Fs_Data(Dataset):
    def __init__(self, file_dir, test_class=2, shot=5, mode='train',HPA=False):
        self.test_class = test_class
        self.mode = mode
        self.shot = shot
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.HPA=HPA
        total_database = load_dataset(file_dir)
        self.test_data = total_database[test_class]
        self.train_data = []

        for i in range(len(total_database)):
            if i != test_class:
                self.train_data.extend(total_database[i])

        if mode == 'train':
            self.dataset = self.train_data
        else:
            self.dataset = self.test_data

        self.total_data = []
        for i in range(len(total_database)):
            self.total_data.append(total_database[i])

    def __len__(self):
        length = int(len(self.dataset))
        return length

    def __getitem__(self, index):
        if self.mode == 'train':
            image_path, label_path = self.train_data[index]
            image, label = load_image(image_path, label_path)
            image, label = self.train_transform(image, label)
            label =label.unsqueeze(0)
            query_index = int(image_path.split('\\')[-2].split('_')[-1])-1

            support_image_list = []
            support_label_list = []
            for i in range(self.shot):
                while len(support_image_list) < self.shot:
                    support_idx = random.randint(0, len(self.train_data)) - 1
                    support_image_path, support_label_path = self.train_data[support_idx]
                    support_index = int(support_image_path.split('\\')[-2].split('_')[-1]) - 1
                    if support_image_path != image_path and support_index ==query_index:
                        image, label = load_image(support_image_path, support_label_path)
                        image, label = self.train_transform(image, label)
                        label = label.unsqueeze(0)

                        support_image_list.append(image)
                        support_label_list.append(label)



        else:
            image_path, label_path = self.dataset[index]
            image, label = load_image(image_path, label_path)
            image, label = self.test_transform(image, label)
            label = label.unsqueeze(0)
            query_index = int(image_path.split('\\')[-2].split('_')[-1]) - 1

            support_image_list = []
            support_label_list = []
            for i in range(self.shot):
                while len(support_image_list) < self.shot:
                    support_idx = random.randint(1, len(self.dataset)) - 1
                    support_image_path, support_label_path = self.dataset[support_idx]
                    support_index = int(support_image_path.split('\\')[-2].split('_')[-1]) - 1
                    if support_image_path != image_path and support_label_path != label_path and support_index ==query_index:
                        support_image, support_label = load_image(support_image_path, support_label_path)
                        support_image, support_label = self.test_transform(support_image, support_label)
                        support_label = support_label.unsqueeze(0)
                        support_image_list.append(support_image)
                        support_label_list.append(support_label)

        if self.HPA:
            return image, label, support_image_list, support_label_list, query_index


        return image, label, support_image_list, support_label_list


class Neu_Data(Dataset):
    def __init__(self, file_dir, test_class=2, shot=5, mode='train'):
        self.test_class = test_class
        self.mode = mode
        self.shot = shot
        self.train_transform = train_transform
        self.test_transform = test_transform
        total_database = load_NEU(file_dir)
        self.test_data = total_database[test_class]
        self.train_data = []

        for i in range(len(total_database)):
            if i != test_class:
                self.train_data.extend(total_database[i])
        if mode == 'train':
            self.dataset = self.train_data
        else:
            self.dataset = self.test_data

        self.total_data = []
        for i in range(len(total_database)):
            self.total_data.append(total_database[i])


    def __len__(self):
        length = int(len(self.dataset))
        return length


    def __getitem__(self, index):
        if self.mode == 'train':
            image_path, label_path = self.train_data[index]
            image, label = load_image(image_path, label_path)
            image, label = self.train_transform(image, label)
            label =label.unsqueeze(0)
            query_class = image_path.split('\\')[-3]

            selected_index =-1
            if query_class == 'AI_Rm' or query_class == 'MT_Uneven' or query_class == 'Steel_Ld' or query_class == "Steel_Am":
                selected_index=0

            if query_class == 'leather' or query_class == 'Steel_Pa' or query_class == 'Rail' or query_class == 'Steel_Sc':
                selected_index = 1

            if query_class == 'Al_Con' or query_class == 'MT_Fray' or query_class == 'MT_Break' or query_class == 'tile':
                selected_index = 2


            support_image_list = []
            support_label_list = []

            for i in range(self.shot):
                while len(support_image_list) < self.shot:
                    support_idx = random.randint(0, len(self.total_data[selected_index])) - 1
                    support_image_path, support_label_path = self.total_data[selected_index][support_idx]
                    support_class = support_image_path.split('\\')[-3]
                    if support_image_path != image_path and support_class == query_class:
                        support_image, support_label = load_image(support_image_path, support_label_path)
                        support_image, support_label = self.train_transform(support_image, support_label)
                        support_label = support_label.unsqueeze(0)
                        support_image_list.append(support_image)
                        support_label_list.append(support_label)

        else:
            image_path, label_path = self.dataset[index]
            image, label = load_image(image_path, label_path)
            image, label = self.test_transform(image, label)
            label = label.unsqueeze(0)
            query_class = image_path.split('\\')[-3]
            support_image_list = []
            support_label_list = []
            for i in range(self.shot):
                while len(support_image_list) < self.shot:
                    support_idx = random.randint(1, len(self.dataset)) - 1
                    support_image_path, support_label_path = self.dataset[support_idx]
                    support_class = support_image_path.split('\\')[-3]
                    if support_image_path != image_path and support_label_path != label_path and support_class == query_class:
                        support_image, support_label = load_image(support_image_path, support_label_path)
                        support_image, support_label = self.test_transform(support_image, support_label)
                        support_label = support_label.unsqueeze(0)
                        support_image_list.append(support_image)
                        support_label_list.append(support_label)


        return image, label, support_image_list, support_label_list

class Neu_Data1(Dataset):
    def __init__(self, file_dir, test_class=2, shot=5, mode='train'):
        self.test_class = test_class
        self.mode = mode
        self.shot = shot
        self.train_transform = train_transform
        self.test_transform = test_transform
        total_database = load_NEU1(file_dir)
        self.test_data = total_database[test_class]
        self.train_data = []

        for i in range(len(total_database)):
            if i != test_class:
                self.train_data.extend(total_database[i])
        if mode == 'train':
            self.dataset = self.train_data
        else:
            self.dataset = self.test_data

        self.total_data = []
        for i in range(len(total_database)):
            self.total_data.append(total_database[i])


    def __len__(self):
        length = int(len(self.dataset))
        return length


    def __getitem__(self, index):
        if self.mode == 'train':
            image_path, label_path = self.train_data[index]
            image, label = load_image(image_path, label_path)
            image, label = self.train_transform(image, label)
            label =label.unsqueeze(0)
            query_class = image_path.split('\\')[-3]

            selected_index =-1
            if query_class == 'AI_Rm' or query_class == 'MT_Uneven' or query_class == 'Al_Con':
                selected_index=0

            if query_class == 'MT_Break' or query_class == 'MT_Fray' or query_class == 'Steel_Ld' :
                selected_index = 1

            if query_class == 'Steel_Am' or query_class == 'Steel_Pa' or query_class == 'Rail':
                selected_index = 2

            if query_class == 'Steel_Sc' or query_class == 'tile' or query_class == 'tile':
                selected_index = 3


            support_image_list = []
            support_label_list = []

            for i in range(self.shot):
                while len(support_image_list) < self.shot:
                    support_idx = random.randint(0, len(self.total_data[selected_index])) - 1
                    support_image_path, support_label_path = self.total_data[selected_index][support_idx]
                    support_class = support_image_path.split('\\')[-3]
                    if support_image_path != image_path and support_class == query_class:
                        support_image, support_label = load_image(support_image_path, support_label_path)
                        support_image, support_label = self.train_transform(support_image, support_label)
                        support_label = support_label.unsqueeze(0)
                        support_image_list.append(support_image)
                        support_label_list.append(support_label)

        else:
            image_path, label_path = self.dataset[index]
            image, label = load_image(image_path, label_path)
            image, label = self.test_transform(image, label)
            label = label.unsqueeze(0)
            query_class = image_path.split('\\')[-3]
            support_image_list = []
            support_label_list = []
            for i in range(self.shot):
                while len(support_image_list) < self.shot:
                    support_idx = random.randint(1, len(self.dataset)) - 1
                    support_image_path, support_label_path = self.dataset[support_idx]
                    support_class = support_image_path.split('\\')[-3]
                    if support_image_path != image_path and support_label_path != label_path and support_class == query_class:
                        support_image, support_label = load_image(support_image_path, support_label_path)
                        support_image, support_label = self.test_transform(support_image, support_label)
                        support_label = support_label.unsqueeze(0)
                        support_image_list.append(support_image)
                        support_label_list.append(support_label)


        return image, label, support_image_list, support_label_list


class FCDD_Data(Dataset):
    def __init__(self, file_dir, test_class=2, shot=5, mode='train'):
        self.test_class = test_class
        self.mode = mode
        self.shot = shot
        self.train_transform = train_transform
        self.test_transform = test_transform
        total_database = load_FSSD(file_dir)
        self.test_data = total_database[test_class]
        self.train_data = []

        for i in range(len(total_database)):
            if i != test_class:
                self.train_data.extend(total_database[i])
        if mode == 'train':
            self.dataset = self.train_data
        else:
            self.dataset = self.test_data

        self.total_data = []
        for i in range(len(total_database)):
            self.total_data.append(total_database[i])


    def __len__(self):
        length = int(len(self.dataset))
        return length


    def __getitem__(self, index):
        if self.mode == 'train':
            image_path, label_path = self.train_data[index]
            image, label = load_image(image_path, label_path)
            image, label = self.train_transform(image, label)
            label =label.unsqueeze(0)
            query_class = image_path.split('\\')[-3]

            selected_index =-1
            if query_class == 'Steel_Am' or query_class == 'Steel_Ia' or query_class == 'Steel_Ld' or query_class == "Steel_Op":
                selected_index = 0

            if query_class == 'Steel_Os' or query_class == 'Steel_Ws' or query_class == 'Steel_Pa' or query_class == 'Steel_Pk':
                selected_index = 1

            if query_class == 'Steel_Ri' or query_class == 'Steel_Rp' or query_class == 'Steel_Sc' or query_class == 'Steel_Se':
                selected_index = 2


            support_image_list = []
            support_label_list = []

            for i in range(self.shot):
                while len(support_image_list) < self.shot:
                    support_idx = random.randint(0, len(self.total_data[selected_index])) - 1
                    support_image_path, support_label_path = self.total_data[selected_index][support_idx]
                    support_class = support_image_path.split('\\')[-3]
                    if support_image_path != image_path and support_class == query_class:
                        support_image, support_label = load_image(support_image_path, support_label_path)
                        support_image, support_label = self.train_transform(support_image, support_label)
                        support_label = support_label.unsqueeze(0)
                        support_image_list.append(support_image)
                        support_label_list.append(support_label)

        else:
            image_path, label_path = self.dataset[index]
            image, label = load_image(image_path, label_path)
            image, label = self.test_transform(image, label)
            label = label.unsqueeze(0)
            query_class = image_path.split('\\')[-3]
            support_image_list = []
            support_label_list = []
            for i in range(self.shot):
                while len(support_image_list) < self.shot:
                    support_idx = random.randint(1, len(self.dataset)) - 1
                    support_image_path, support_label_path = self.dataset[support_idx]
                    support_class = support_image_path.split('\\')[-3]
                    if support_image_path != image_path and support_label_path != label_path and support_class == query_class:
                        support_image, support_label = load_image(support_image_path, support_label_path)
                        support_image, support_label = self.test_transform(support_image, support_label)
                        support_label = support_label.unsqueeze(0)
                        support_image_list.append(support_image)
                        support_label_list.append(support_label)


        return image, label, support_image_list, support_label_list


if __name__ =='__main__':
    data = Neu_Data(file_dir=r'D:\IMSN-YHM\dataset\MSD-Seg2', test_class=1, mode='test')
    dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)
    import matplotlib.pyplot as plt

    for item in dataloader:
        image, label, support_image_list, support_label_list = item
        plt.subplot(221)
        plt.imshow(denormalize(image[0].cpu().data.numpy()))
        plt.subplot(222)
        plt.imshow(label[0][0].cpu().data.numpy())
        plt.subplot(223)
        plt.imshow(denormalize(support_image_list[0][0].cpu().data.numpy()))
        plt.subplot(224)
        plt.imshow(support_label_list[0][0][0].cpu().data.numpy())
        plt.show()
        # print(image.shape)
        # print(label.shape)




