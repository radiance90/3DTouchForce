import os.path
import random
import cv2
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
from torchvision import transforms
from scipy import ndimage


class ForceDataset(Dataset):

    def __init__(self, data_list, phase):
        self.data_list = data_list
        self.phase = phase

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path1 = os.path.join(item[0], item[1]).replace('/mnt/data1/yjy/TouchForce/','../')
        img_path2 = os.path.join(item[0], item[2]).replace('/mnt/data1/yjy/TouchForce/','../')

        img1 = np.load(img_path1)
        img2 = np.load(img_path2)
        
        img1[img1<0] = 0
        img2[img2<0] = 0
        max_val = np.max(img1)
        img1 /= max_val
        img2 /= max_val

        location_x = int(item[7] / 720 * 18)
        location_y = int(item[6] / 1600 * 32)

        force_x = np.float32(item[3])
        force_y = np.float32(item[4])
        force_z = np.float32(item[5])

        action = item[10]

        if self.phase == 'train':
            location_x += random.randint(-1, 1)
            location_y += random.randint(-1, 1)

        img1 = ndimage.shift(img1, (16-location_y, 9-location_x), mode='constant', cval=0)
        img2 = ndimage.shift(img2, (16-location_y, 9-location_x), mode='constant', cval=0)

        img1 = np.pad(img1,((0,),(7,)))[None]
        img2 = np.pad(img2,((0,),(7,)))[None]

        if self.phase == 'train' and random.random()>0.5:
            img1 = np.flip(img1, axis=2)
            img2 = np.flip(img2, axis=2)
            force_x = np.float32(force_x * -1)

        img = np.concatenate((img1, img2)).astype("float32")

        return img, force_x, force_y, force_z, action




