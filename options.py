
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='./swin_base_patch4_window12_384_22k.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='3', help='train use gpu')

parser.add_argument('--rgb_root', type=str, default='../dataset/usod/Train/V/', help='the training RGB images root')
parser.add_argument('--rgb1_root', type=str, default='../dataset/usod/Train/VG/', help='the training Gamma images root')
parser.add_argument('--depth_root', type=str, default='../dataset/usod/Train/D/', help='the training Depth images root')
parser.add_argument('--gt_root', type=str, default='../dataset/usod/Train/GT/', help='the training GT images root')
parser.add_argument('--edge_root', type=str, default='../dataset/usod/Train/E/', help='the training Edge images root')

parser.add_argument('--val_rgb_root', type=str, default='../dataset/usod/Test/V/', help='the validation RGB images root')
parser.add_argument('--val_rgb1_root', type=str, default='../dataset/usod/Test/VG/', help='the validation Gamma images root')
parser.add_argument('--val_depth_root', type=str, default='../dataset/usod/Test/D/', help='the validation Depth images root')
parser.add_argument('--val_gt_root', type=str, default='../dataset/usod/Test/GT/', help='the validation GT images root')

parser.add_argument('--save_path', type=str, default='./cpts/', help='the path to save models and logs')

opt = parser.parse_args()
