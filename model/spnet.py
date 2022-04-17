from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F

from copy import deepcopy
from collections import OrderedDict

import model.resnet as models
from model.asgnet import Model
from model.module.decoder import build_decoder
from model.module.ASPP import ASPP

# Masked Average Pooling
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

class SPNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.zoom_factor = args.zoom_factor
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.shot = args.shot
        self.train_iter = args.train_iter
        self.eval_iter = args.eval_iter
        self.pyramid = args.pyramid

        asg_sd = torch.load('./initmodel/s0_1s.pth')['state_dict']
        new_sd = OrderedDict()
        for k, v in asg_sd.items():
            new_sd[k.replace('module.', '')] = v

        asg = Model(args)
        asg.load_state_dict(new_sd)

        self.layer0 = asg.layer0
        self.layer1 = asg.layer1
        self.layer2 = asg.layer2
        self.layer3 = asg.layer3
        self.layer4 = asg.layer4
        self.down_conv = asg.down_conv
        self.cls = asg.cls
        
        # reduce_dim = 256
        # fea_dim = 1024 + 512

    def sp_center_iter(self, supp_feat, supp_mask, sp_init_center, n_iter):
        '''
        :param supp_feat: A Tensor of support feature, (C, H, W)
        :param supp_mask: A Tensor of support mask, (1, H, W)
        :param sp_init_center: A Tensor of initial sp center, (C + xy, num_sp)
        :param n_iter: The number of iterations
        :return: sp_center: The centroid of superpixels (prototypes)
        '''

        c_xy, num_sp = sp_init_center.size()
        _, h, w = supp_feat.size()
        h_coords = torch.arange(h).view(h, 1).contiguous().repeat(1, w).unsqueeze(0).float().cuda()
        w_coords = torch.arange(w).repeat(h, 1).unsqueeze(0).float().cuda()
        supp_feat = torch.cat([supp_feat, h_coords, w_coords], 0)
        supp_feat_roi = supp_feat[:, (supp_mask == 1).squeeze()]  # (C + xy) x num_roi

        num_roi = supp_feat_roi.size(1)
        supp_feat_roi_rep = supp_feat_roi.unsqueeze(-1).repeat(1, 1, num_sp)
        sp_center = torch.zeros_like(sp_init_center).cuda()  # (C + xy) x num_sp

        for i in range(n_iter):
            # Compute association between each pixel in RoI and superpixel
            if i == 0:
                sp_center_rep = sp_init_center.unsqueeze(1).repeat(1, num_roi, 1)
            else:
                sp_center_rep = sp_center.unsqueeze(1).repeat(1, num_roi, 1)
            assert supp_feat_roi_rep.shape == sp_center_rep.shape  # (C + xy) x num_roi x num_sp
            dist = torch.pow(supp_feat_roi_rep - sp_center_rep, 2.0)
            feat_dist = dist[:-2, :, :].sum(0)
            spat_dist = dist[-2:, :, :].sum(0)
            total_dist = torch.pow(feat_dist + spat_dist / 100, 0.5)
            p2sp_assoc = torch.neg(total_dist).exp()
            p2sp_assoc = p2sp_assoc / (p2sp_assoc.sum(0, keepdim=True))  # num_roi x num_sp

            sp_center = supp_feat_roi_rep * p2sp_assoc.unsqueeze(0)  # (C + xy) x num_roi x num_sp
            sp_center = sp_center.sum(1)

        return sp_center[:-2, :]

    def sp_center(self, s_seed, supp_feat, supp_mask):
        sp_init_center = supp_feat[:, s_seed[:, 0], s_seed[:, 1]]                           # c x num_sp (sp_seed)
        sp_init_center = torch.cat([sp_init_center, s_seed.transpose(1, 0).float()], dim=0) # (c + xy) x num_sp
        return self.sp_center_iter(
            supp_feat, supp_mask, sp_init_center,
            n_iter=self.train_iter if self.training else self.eval_iter
        )

    def forward(self, x,
        s_x=torch.FloatTensor(1, 1, 3, 473, 473).cuda(),
        s_y=torch.FloatTensor(1, 1, 473, 473).cuda(),
        s_fg_seed=None, s_bg_seed=None, y=None):

        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_conv(query_feat)

        # Support Feature
        supp_feat_list = []
        mask_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                mask_list.append(mask)  # Get all the downsampled mask

            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_conv(supp_feat)
            supp_feat_list.append(supp_feat)  # shot x [bs x 256 x h x w]

        ################## Adaptive Superpixel Clustering ######################

        bs, _, max_num_sp, _ = s_fg_seed.size()  # bs x shot x max_num_sp x 2
        all_fg_center, all_bg_center = [], []
        for bs_ in range(bs):
            sp_fg_center_list, sp_bg_center_list = [], []
            for shot_ in range(self.shot):
                with torch.no_grad():
                    supp_feat_ = supp_feat_list[shot_][bs_, :, :, :]  # c x h x w

                    supp_fg_mask = mask_list[shot_][bs_, :, :, :]       # 1 x h x w
                    supp_bg_mask = (supp_fg_mask == 0).float()
                    s_fg_seed_ = s_fg_seed[bs_, shot_, :, :]        # max_num_sp x 2
                    s_bg_seed_ = s_bg_seed[bs_, shot_, :, :]        # max_num_sp x 2
                    num_fg_sp = max(len(torch.nonzero(s_fg_seed_[:, 0])), len(torch.nonzero(s_fg_seed_[:, 1])))
                    num_bg_sp = max(len(torch.nonzero(s_bg_seed_[:, 0])), len(torch.nonzero(s_bg_seed_[:, 1])))
                    s_fg_seed_ = s_fg_seed_[:num_fg_sp, :]          # num_fg_sp x 2
                    s_bg_seed_ = s_bg_seed_[:num_bg_sp, :]          # num_bg_sp x 2

                    # if num_sp == 0 or 1, use the Masked Average Pooling instead
                    if (num_fg_sp == 0) or (num_fg_sp == 1):
                        supp_fg_proto = Weighted_GAP(supp_feat_.unsqueeze(0), supp_fg_mask.unsqueeze(0))    # 1 x c x 1 x 1
                        sp_fg_center_list.append(supp_fg_proto.squeeze().unsqueeze(-1))                     # c x 1
                    else:
                        sp_fg_center_list.append(self.sp_center(
                            s_seed=s_fg_seed_,
                            supp_feat=supp_feat_,
                            supp_mask=supp_fg_mask
                        ))
                    if (num_bg_sp == 0) or (num_bg_sp == 1):
                        supp_bg_proto = Weighted_GAP(supp_feat_.unsqueeze(0), supp_bg_mask.unsqueeze(0))    # 1 x c x 1 x 1
                        sp_bg_center_list.append(supp_bg_proto.squeeze().unsqueeze(-1))                     # c x 1
                    else:
                        sp_bg_center_list.append(self.sp_center(
                            s_seed=s_bg_seed_,
                            supp_feat=supp_feat_,
                            supp_mask=supp_fg_mask
                        ))

            sp_fg_center = torch.cat(sp_fg_center_list, dim=1)   # c x num_sp_all (collected from all shots)
            sp_bg_center = torch.cat(sp_bg_center_list, dim=1)   # c x num_sp_all (collected from all shots)
            all_fg_center.append(sp_fg_center)
            all_bg_center.append(sp_bg_center)

        # import pdb; pdb.set_trace()

        return all_fg_center, all_bg_center, supp_feat_list, mask_list
        