r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
from models.backbone import *
from torch.distributions import Normal, kl_divergence
import math

def likelihood(sample_mean, mean, log_var):
    dist = Normal(mean, log_var.exp())
    prob = dist.log_prob(sample_mean).exp()
    return torch.mean(prob, dim=1)



def distribution_discrepancy_loss(mean_1, log_var_1, mean_2, log_var_2):
    distance = kl_loss(mean_1, log_var_1, mean_2, log_var_2)
    #print(distance)
    simi = 1/(1+distance)
    return simi


class Sample(nn.Module):
    def __init__(self):
        super(Sample, self).__init__()
    def forward(self,z_mean,z_log_var):
        epsilon=torch.randn(z_mean.shape)
        epsilon=epsilon.to(z_mean.device)
        return z_mean+(z_log_var/2).exp()*epsilon


def kl_loss(mean_1, log_var_1,mean_2, log_var_2,reduce=False):
    loss = 1 + (log_var_1-log_var_2) - (torch.exp(log_var_1)/torch.exp(log_var_2))-((mean_1-mean_2)**2/torch.exp(log_var_2))
    if reduce:
        return -0.5 * torch.mean(loss)
    else:
        return -0.5 * torch.mean(loss, dim=1)





class Conv_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class VAE(nn.Module):
    def __init__(self, input_channel, latent_dim):
        super(VAE, self).__init__()
        self.layer1 = Conv_module(in_channels=input_channel, out_channels=1024)
        self.layer2 = Conv_module(in_channels=1024, out_channels=512)
        self.layer3 = Conv_module(in_channels=512, out_channels=2*latent_dim)


    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        mean, log_var = out3.chunk(2, dim=1) # chunk是拆分，dim=1是在一维上拆分
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, input_channel):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=3, dilation=1,  padding=1),   #fc6
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Conv2d(128, 2, kernel_size=1, padding=1)

    def forward(self, input_):
        out1 = self.layer1(input_)
        output = self.output_layer(out1)
        output = F.upsample(output, size=(256, 256), mode='bilinear')

        return output


class DSVP(nn.Module):
    def __init__(self, backbone):
        super(DSVP, self).__init__()
        self.backbone_type = backbone
        if backbone == 'vgg16':
            self.backbone = VGG()
            self.channels = 960
        elif backbone == 'resnet50':
            self.backbone = Resnet50()
            self.channels = 1856
        elif backbone == 'resnet101':
            self.backbone = Resnet101()
            self.channels = 1856

        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.backbone.eval()
        self.prior_network = VAE(input_channel=self.channels, latent_dim=self.channels)
        self.posterior_network = VAE(input_channel=self.channels, latent_dim=self.channels)
        self.back_decoder = Decoder(input_channel=1*self.channels+2)
        self.fore_decoder = self.back_decoder
        #self.decoder = Decoder(input_channel=2 * self.channels+2)
        self.cross_entropy_loss = nn.CrossEntropyLoss()


    def forward(self, query_img, support_img, support_mask):
        with torch.no_grad():
            query_feats = self.backbone(query_img)
            support_feats = self.backbone(support_img)

        ############# similarity #############
        forground_similarity_map = self.attention_map(support_feats, support_mask, query_feats)
        background_similarity_map = self.attention_map(support_feats, 1 - support_mask, query_feats)

        ############# support ############
        support_forground_prototype = self.mask_feature(support_feats, support_mask.clone())
        support_background_prototype = self.mask_feature(support_feats, 1 - support_mask.clone())

        support_forground_mu, support_forground_log_var = self.prior_network(support_forground_prototype)
        support_background_mu, support_background_log_var = self.prior_network(support_background_prototype)

        support_forground_feature = Sample()(support_forground_mu, support_forground_log_var).repeat(1, 1, 64, 64)
        support_background_feature = Sample()(support_background_mu, support_background_log_var).repeat(1, 1, 64, 64)

        ############# query ############

        query_mu, query_log_var = self.posterior_network(query_feats)

        ############# likelihood ############
        forground_divergence = kl_loss(query_mu, query_log_var, support_forground_mu, support_forground_log_var,
                                       reduce=False).unsqueeze(1)
        background_divergence = kl_loss(query_mu, query_log_var, support_background_mu, support_background_log_var,
                                        reduce=False).unsqueeze(1)


        background_likelihood = (background_divergence.max() - background_divergence) / (background_divergence.max() - background_divergence.min())

        forground_similarity_map = (forground_similarity_map - forground_similarity_map.min()) / (
                forground_similarity_map.max() - forground_similarity_map.min())

        background_similarity_map = (background_similarity_map - background_similarity_map.min()) / (
                background_similarity_map.max() - background_similarity_map.min())

        forground_likelihood = 1 - background_likelihood



        ############# decoder ############
        # forground_output_feature = torch.cat(
        #     [query_feats, support_forground_feature],
        #     dim=1)*forground_similarity_map.unsqueeze(1)
        #
        # background_output_feature = torch.cat(
        #     [query_feats, support_background_feature],
        #     dim=1)*background_similarity_map.unsqueeze(1)
        forground_output_feature = torch.cat(
            [query_feats, forground_similarity_map.unsqueeze(1), forground_likelihood],
            dim=1)

        background_output_feature = torch.cat(
            [query_feats, background_similarity_map.unsqueeze(1), background_likelihood],
            dim=1)


        forground_output = self.fore_decoder(forground_output_feature)
        background_output = self.back_decoder(background_output_feature)

        result ={"output":[forground_output, background_output]}

        result["similarity"] = [forground_similarity_map, background_similarity_map]
        result["divergence"] = [forground_likelihood, background_likelihood]


        return result

    def loss(self, query_img, query_mask, support_img, support_mask):
        with torch.no_grad():
            query_feats = self.backbone(query_img)
            support_feats = self.backbone(support_img)

        ############# similarity #############
        forground_similarity_map = self.attention_map(support_feats, support_mask, query_feats)
        background_similarity_map = self.attention_map(support_feats, 1-support_mask, query_feats)


        ############# support ############
        support_forground_prototype = self.mask_feature(support_feats, support_mask.clone())
        support_background_prototype = self.mask_feature(support_feats, 1-support_mask.clone())

        support_forground_mu, support_forground_log_var = self.prior_network(support_forground_prototype)
        support_background_mu, support_background_log_var = self.prior_network(support_background_prototype)

        support_forground_feature = Sample()(support_forground_mu, support_forground_log_var).repeat(1, 1, 64, 64)
        support_background_feature = Sample()(support_background_mu, support_background_log_var).repeat(1, 1, 64, 64)


        ############# query ############

        query_mu, query_log_var = self.posterior_network(query_feats)

        ############# likelihood ############
        forground_divergence = kl_loss(query_mu, query_log_var, support_forground_mu, support_forground_log_var,
                                       reduce=False).unsqueeze(1)
        background_divergence = kl_loss(query_mu, query_log_var, support_background_mu, support_background_log_var,
                                        reduce=False).unsqueeze(1)

        forground_D_kl_loss = torch.sum(self.mask_feature(forground_divergence, query_mask.clone()))
        background_D_kl_loss = torch.sum(self.mask_feature(background_divergence, 1-query_mask.clone()))

        #############


        background_likelihood = (background_divergence.max() - background_divergence) / (
                    background_divergence.max() - background_divergence.min())

        forground_similarity_map = (forground_similarity_map - forground_similarity_map.min()) / (
                forground_similarity_map.max() - forground_similarity_map.min())

        background_similarity_map = (background_similarity_map - background_similarity_map.min()) / (
                background_similarity_map.max() - background_similarity_map.min())


        ############# decoder ############
        forground_output_feature = torch.cat(
            [query_feats, forground_similarity_map.unsqueeze(1), forground_likelihood],
            dim=1)

        background_output_feature = torch.cat(
            [query_feats, background_similarity_map.unsqueeze(1), background_likelihood],
            dim=1)


        forground_output = self.fore_decoder(forground_output_feature)
        background_output = self.back_decoder(background_output_feature)

        ############# loss  ############

        D_kl_loss = forground_D_kl_loss + background_D_kl_loss

        b, c, w, h = query_mask.size()
        forground_output = forground_output.permute(0, 2, 3, 1).contiguous().view(b * w * h, 2)
        background_output = background_output.permute(0, 2, 3, 1).contiguous().view(b * w * h, 2)

        query_mask = query_mask.view(-1)

        forground_loss_bce_seg = self.cross_entropy_loss(forground_output, query_mask.long())
        background_loss_bce_seg = self.cross_entropy_loss(background_output, 1-query_mask.long())
        loss_bce_seg = forground_loss_bce_seg+background_loss_bce_seg

        return D_kl_loss, loss_bce_seg



    def mask_feature(self, features, support_mask):
        _, channel, mask_w, mask_h = features.size()

        support_mask = nn.functional.interpolate(torch.tensor(support_mask, dtype=torch.float), size=(mask_w, mask_h))
        vec_pos = torch.sum(torch.sum(features * support_mask, dim=3), dim=2) / torch.sum(support_mask)
        vec_pos = vec_pos.unsqueeze(-1).unsqueeze(-1)
        return vec_pos

    def get_pred(self, query_img, support_img, support_mask):
        result = self(query_img, support_img, support_mask)
        pred_forground = result["output"][0]
        pred_background = result["output"][1]

        forground_out_softmax = F.softmax(pred_forground, dim=1).squeeze()
        values, forground_pred = torch.max(forground_out_softmax, dim=0)

        background_out_softmax = F.softmax(pred_background, dim=1).squeeze()
        values, background_pred = torch.max(background_out_softmax, dim=0)

        forground_similarity_map, background_similarity_map = result["similarity"]
        forground_likelihood, background_likelihood = result["divergence"]
        return forground_pred, background_pred, forground_similarity_map, background_similarity_map, forground_likelihood, background_likelihood

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from lea rning data statistics with exponential averaging

    def attention_map(self, support_feature, support_mask, query_feature):
        eps = 1e-5
        bsz, ch, hb, wb = support_feature.size()
        support_feature = support_feature.view(bsz, ch, -1)
        support_feature = support_feature / (support_feature.norm(dim=1, p=2, keepdim=True) + eps)

        bsz, ch, ha, wa = query_feature.size()
        query_feature = query_feature.view(bsz, ch, -1)
        query_feature = query_feature / (query_feature.norm(dim=1, p=2, keepdim=True) + eps)

        support_mask = nn.functional.interpolate(torch.tensor(support_mask, dtype=torch.float), size=(ha, wa))

        corr = torch.bmm(query_feature.transpose(1, 2), support_feature).view(bsz, ha, wa, hb, wb)
        corr = corr.clamp(min=0)
        corr = corr.view(bsz, ha, wa, -1)
        support_mask = support_mask.view(bsz, 1, -1).view(bsz, 1, 1, -1)
        corr = torch.softmax(corr, dim=-1)
        corr = torch.sum(torch.mul(corr, support_mask), dim=-1)
        return corr

if __name__ == '__main__':
    mean_1 = torch.rand(1, 256, 1, 1).cuda()
    log_var_1 = torch.rand(1, 256,1,1).cuda()
    mean_2 = torch.rand(1, 256,1,1).cuda()
    log_var_2 = torch.rand(1, 256,1,1).cuda()
    loss = likelihood(mean_2, mean_1, log_var_1)
    print(loss)






    # model = DSVP(backbone='vgg16').cuda()
    # query_img = torch.rand(8, 3, 256, 256).cuda()
    # query_mask = torch.rand(8, 1, 256, 256).cuda()
    # support_img = torch.rand(8, 3, 256, 256).cuda()
    # support_mask = torch.rand(8, 1, 256, 256).cuda()
    #
    # output = model.loss(query_img, query_mask, support_img, support_mask)
    # print(output[0])
    # print(output[1])

