
import cv2
import random
import logging
import argparse
import numpy as np
from addict import Dict
from tensorboardX import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util import dataset, transform, config
from util.util import AverageMeter
from model.spnet import SPNet

CONFIG = Dict(
    # default settings
    seed=42,
    device='cpu',
    # dataset
    split=0,
    train_h=473,
    train_w=473,
    # transform
    scale_min=0.9,
    rotate_min=-10,
    rotate_max=10,
    zoom_factor=8,
    ignore_label=255,
    padding_label=255
)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def init_seed(manual_seed=42):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

def main():

    args = get_parser()
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0

    if args.manual_seed is not None:
        init_seed(CONFIG.seed)
    
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    
    # device = 'cpu' if torch.cuda.is_available() else 'cuda'
    # model = SPNet(args).to(device)
    model = SPNet(args).cuda()

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]
    train_transform = [
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    train_data = dataset.SemData_SP(
        split=args.split,
        shot=args.shot,
        max_sp=args.max_sp,
        data_root=args.data_root,
        data_list=args.train_list,
        transform=train_transform,
        mode='train',
        use_coco=args.use_coco,
        use_split_coco=args.use_split_coco
    )

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    # if args.evaluate:
    #     if args.resized_val:
    #         val_transform = transform.Compose([
    #             transform.Resize(size=args.val_size),
    #             transform.ToTensor(),
    #             transform.Normalize(mean=mean, std=std)])    
    #     else:
    #         val_transform = transform.Compose([
    #             transform.test_Resize(size=args.val_size),
    #             transform.ToTensor(),
    #             transform.Normalize(mean=mean, std=std)])           
    #     val_data = dataset.SemData_SP(
    #         split=args.split,
    #         shot=args.shot,
    #         max_sp=args.max_sp,
    #         data_root=args.data_root,
    #         data_list=args.val_list,
    #         transform=val_transform,
    #         mode='val',
    #         use_coco=args.use_coco,
    #         use_split_coco=args.use_split_coco
    #     )
    #     val_sampler = None
    #     val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    max_iou = 0.0
    filename = 'SPNet.pth'

    image, label, s_x, s_y, s_fg_seed, s_bg_seed, subcls_list = iter(train_loader).next()
    s_x = s_x.cuda(non_blocking=True)
    s_y = s_y.cuda(non_blocking=True)
    image = image.cuda(non_blocking=True)
    label = label.cuda(non_blocking=True)
    s_fg_seed = s_fg_seed.cuda(non_blocking=True)
    s_bg_seed = s_bg_seed.cuda(non_blocking=True)
    
    cuda2cpu = lambda lst: [i.cpu() for i in lst]
    pred = list(map(cuda2cpu, model(image, s_x, s_y, s_fg_seed, s_bg_seed, label)))
    dump_dict = dict(
        fg_centers=pred[0],
        bg_centers=pred[1],
        sp_feats=pred[2],
        sp_masks=pred[3],
        s_x=s_x.cpu(),
        s_y=s_y.cpu(),
    )
    torch.save(dump_dict, 'fixed_pred_down.obj')

if __name__ == '__main__':
    main()