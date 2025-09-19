import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance

# Reference from BBSNet, Thanks!!!

# several data augumentation strategies
def cv_random_flip(img,img1, label, depth):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        #edge = edge.transpose(Image.FLIP_LEFT_RIGHT)

    return img,img1, label, depth#, edge


def randomCrop(image,image1, label, depth):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region),image1.crop(random_region), label.crop(random_region), depth.crop(random_region)#, edge.crop(random_region)


def randomRotation(image,image1, label, depth):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        image1 = image1.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)

    return image,image1, label, depth


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root,image1_root, gt_root, depth_root,  trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')or f.endswith('.png')]
        self.images1 = [image1_root + f for f in os.listdir(image1_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png') or f.endswith('.jpg')]

        self.images = sorted(self.images)
        self.images1 = sorted(self.images1)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)

        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        image1 = self.rgb_loader(self.images1[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.binary_loader(self.depths[index]) # RGBD
        #depth = self.rgb_loader(self.depths[index])  # RGBT


        image,image1, gt, depth = cv_random_flip(image,image1, gt, depth, )
        image,image1, gt, depth = randomCrop(image,image1, gt, depth)
        image,image1, gt, depth = randomRotation(image,image1, gt, depth)
        #image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        image1 = self.img_transform(image1)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)


        return image,image1, gt, depth

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        images1 = []
        gts = []
        depths = []

        for img_path, img1_path,gt_path, depth_path in zip(self.images,self.images1, self.gts, self.depths):
            img = Image.open(img_path)
            img1 = Image.open(img1_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)

            if img.size == gt.size and gt.size == depth.size and img1.size == img.size:
                images.append(img_path)
                images1.append(img1_path)
                gts.append(gt_path)
                depths.append(depth_path)

        self.images = images
        self.images1 = images1
        self.gts = gts
        self.depths = depths


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img,img1, gt, depth):
        assert img.size == gt.size and gt.size == depth.size and edge.size == img.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR),img1.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), \
                   depth.resize((w, h),Image.NEAREST)
        else:
            return img,img1, gt, depth

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root,image1_root, gt_root, depth_root,  batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    dataset = SalObjDataset(image_root,image1_root, gt_root, depth_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root,image1_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')or f.endswith('.png')]
        self.images1 = [image1_root + f for f in os.listdir(image1_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')or f.endswith('.jpg')]

        self.images = sorted(self.images)
        self.images1 = sorted(self.images1)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        image1 = self.rgb_loader(self.images1[self.index])
        image1 = self.transform(image1).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        depth = self.binary_loader(self.depths[self.index]) # RGBD
        #depth = self.rgb_loader(self.depths[self.index]) # RGBT
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image,image1, gt, depth, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

