
import os
import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.FBSNet import SwinTransformer,FBSNet
from torchvision.utils import make_grid
from data import get_loader, test_dataset
import logging
import torch.backends.cudnn as cudnn
from options import opt
import yaml
import torch.nn as nn
import pytorch_losses

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)
cudnn.benchmark = True

image_root = opt.rgb_root
image1_root = opt.rgb1_root
gt_root = opt.gt_root
depth_root = opt.depth_root
edge_root = opt.edge_root

val_image_root = opt.val_rgb_root
val_image1_root = opt.val_rgb1_root
val_gt_root = opt.val_gt_root
val_depth_root = opt.val_depth_root
save_path = opt.save_path


model = FBSNet()

num_parms = 0
if (opt.load is not None):
    model.load_pre(opt.load)#-------------------------------------------------------
    print('load model from ', opt.load)

model.cuda()
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0)
# set the path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root,image1_root, gt_root,depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(val_image_root,val_image1_root, val_gt_root, val_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()
ECE = nn.BCELoss()

def Hybrid_Loss(pred, target, reduction='mean'):
    pred = torch.sigmoid(pred)

    #BCE LOSS
    bce_loss = nn.BCELoss()
    bce_out = bce_loss(pred, target)

    #IOU LOSS
    iou_loss = pytorch_losses.IOU(reduction=reduction)
    iou_out = iou_loss(pred, target)

    #SSIM LOSS
    ssim_loss = pytorch_losses.SSIM(window_size=11)
    ssim_out = ssim_loss(pred, target)

    losses = bce_out + iou_out + ssim_out

    return  losses

step = 0
best_mae = 1
best_epoch = 0


def train(train_loader, model, optimizer, epoch, save_path,scheduler):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images,images1, gts, depth) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            images1 = images1.cuda()
            gts = gts.cuda()
            depth = depth.repeat(1,3,1,1).cuda()

            s,s1,s2,s3,s4= model(images,depth,images1)

            sal_loss = Hybrid_Loss(s, gts)
            loss1 = Hybrid_Loss(s1, gts)
            loss2 = Hybrid_Loss(s2, gts)
            loss3 = Hybrid_Loss(s3, gts)
            loss4 = Hybrid_Loss(s4, gts)

            loss = sal_loss + loss1 + loss2 + loss3 + loss4
            loss.backward()

            optimizer.step()
            if i==1:
                scheduler.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f} ||edge_loss:{:4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], sal_loss.data, sal_loss.data))

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        if (epoch) % 20 == 0:
            torch.save(model.state_dict(), save_path + 'FBSNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'FBSNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image,image1, gt, depth, name, img_for_post = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            image1 = image1.cuda()
            depth = depth.repeat(1,3,1,1).cuda()
            res,s1,s2,s3,s4 = model(image,depth,image1)
            
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'FBSNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        train(train_loader, model, optimizer, epoch, save_path,scheduler=schedule)
        test(test_loader, model, epoch, save_path)
