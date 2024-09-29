# coding:utf-8
import torchvision.transforms.functional as TF
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import glob
from natsort import natsorted
import argparse
import cv2
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler

class Fusion_dataset(Dataset):
    def __init__(self, data_dir, sub_dir, transform):
        super(Fusion_dataset, self).__init__()
        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'ir')))
        self.length = len(self.img_names)
        self.transform = transform  #

    def __getitem__(self, index):
        img_name = self.img_names[index]
        vis = cv2.imread(os.path.join(self.root_dir, 'vis_gray', img_name), 0)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        ir = cv2.imread(os.path.join(self.root_dir, 'ir', img_name), 0)
        ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        vis = Image.fromarray(vis)
        ir = Image.fromarray(ir)

        vis = self.transform(vis)
        ir = self.transform(ir)

        return ir, vis, img_name

    def __len__(self):
        return self.length


