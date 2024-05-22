import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.models.vgg import vgg16, vgg19,vgg19_bn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101,wide_resnet50_2, wide_resnet101_2
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import mobilenet_v2

class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        resnet = resnet34(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]


    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)

        f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)
        return f_


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg = vgg16(True)
        layers = vgg.features
        # print(layers)
        self.layer1 = layers[:5]
        self.layer2 = layers[5:10]
        self.layer3 = layers[10:17]
        self.layer4 = layers[17:24]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)

        f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)
        return f_


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        resnet = resnet18(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]


    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)

        f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)
        return f_


class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        resnet = resnet34(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]


    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)

        f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)
        return f_


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        resnet = resnet50(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]


    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)
   

        f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)
        return f_


class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        resnet = resnet101(True)

        modules = list(resnet.children())
        self.block1 = nn.Sequential(*modules[0:4])
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]


    def forward(self, input_):
        out1 = self.block1(input_)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
 
        f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)


        f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)
        f_ = F.normalize(f_)
        return f_



class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg = vgg19(True)
        layers = vgg.features
        # print(layers)
        self.layer1 = layers[:5]
        self.layer2 = layers[5:10]
        self.layer3 = layers[10:19]
        self.layer4 = layers[19:28]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)

        out3 = self.layer3(out2)

        out4 = self.layer4(out3)
        f1_ = F.interpolate(out1, size=(64, 64), mode='bilinear', align_corners=True)
        f2_ = F.interpolate(out2, size=(64, 64), mode='bilinear', align_corners=True)
        f3_ = F.interpolate(out3, size=(64, 64), mode='bilinear', align_corners=True)
        f4_ = F.interpolate(out4, size=(64, 64), mode='bilinear', align_corners=True)

        f_ = torch.cat([f1_, f2_, f3_, f4_], dim=1)

        return f_





if __name__=="__main__":
    model = VGG19().cuda()
    tensor = torch.randn(1, 3, 256, 256).cuda()
    import time
    for i in range(10):
        t1 = time.time()
        output = model(tensor)
        t2 = time.time()
        print(t2-t1)

    print(output.shape)