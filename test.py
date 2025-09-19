
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.FBSNet import FBSNet
from data import test_dataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='../dataset/usod/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model = FBSNet()

model.load_state_dict(torch.load("./SwinNet_epoch_best.pth"))

model.cuda()
model.eval()
fps = 0


test_datasets = ['Test']
for dataset in test_datasets:
    time_s = time.time()
    sal_save_path = './save/' + dataset + '/'
    if not os.path.exists(sal_save_path):
        os.makedirs(sal_save_path)
    image_root = dataset_path + dataset + '/V/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/D/'
    vg_root = dataset_path + dataset + '/VG/'
    test_loader = test_dataset(image_root,vg_root, gt_root, depth_root, opt.testsize)
    nums = test_loader.size
    for i in range(test_loader.size):
        image,vg, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        vg = vg.cuda()
        depth = depth.repeat(1,3,1,1).cuda()

        res = model(image,depth,vg)
        res,*_ = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)

        res = res.sigmoid().data.cpu().numpy().squeeze()

        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        print('save img to: ', sal_save_path + name)
        cv2.imwrite(sal_save_path + name, res * 255)


