import os
import os.path
import random
import math
import errno
import glob
import functools

from data import common

import numpy as np

import torch
import torch.utils.data as data
import data.utils as Utils
from torchvision import transforms

class Dataset4AIR(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        """
        dir_data:
        |
        |-LR
        |  |-*.npy
        |-HR
        |  |-*.npy
        """
        self.dataroot = args["dataroot"]
        if args["storage"] == 'file':
            self.storage = Utils.FileStorage(self.dataroot)
        elif args["storage"] == 'ceph':
            self.storage = Utils.CephStorage(self.dataroot)

        self.args = args
        self.train = train
        self.benchmark=benchmark
        self.name = '4air'
        #print(args)
        #self.dir_data=args.dir_data #to change!
        self.dataroot_L=args["dataroot_L"]
        self.dataroot_H=args["dataroot_H"]
        # define shape
        self.lr_shape = self.args['lr_shape']
        self.hr_shape = self.args['hr_shape']


        if self.args['inference']:
            # LR
            print("INFERENCE MODE: Loading LR files: {}".format(os.path.join(self.dataroot_L,'*.npy')))
            self.LR_path = self.storage.objectslist(self.dataroot_L) #glob.glob(os.path.join(self.dataroot_L,'*.npy'))
            print("len(LR files): {}".format(len(self.LR_path)))
            # Set index
            self.files = [os.path.basename(x) for x in self.LR_path]
        else:
            # LR
            print("Loading LR files: {}".format(os.path.join(self.dataroot_L,'*.npy')))
            self.LR_path = self.storage.objectslist(self.dataroot_L) #glob.glob(os.path.join(self.dataroot_L,'*.npy'))
            print("len(LR files): {}".format(len(self.LR_path)))
            # Set index
            self.files = [os.path.basename(x) for x in self.LR_path]
            #self.idxs = sorted([int(os.path.basename(x)[:-4]) for x in self.LR_path])
            #self.idxs = [int(x) for x in self.idxs]
            # HR
            print("Loading HR files: {}".format(os.path.join(self.dataroot_H,'*.npy')))
            self.HR_path = self.storage.objectslist(self.dataroot_H) #glob.glob(os.path.join(self.dataroot_H,'*.npy'))
            print("len(HR files): {}".format(len(self.HR_path)))
            # shuffle index
            #random.seed(args["seed"])
            #self.idxs = list(range(len(self.LR_path)))
            #random.shuffle(self.idxs)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        #filename = os.path.split(self.filelist[idx])[-1]
        #filename, _ = os.path.splitext(filename)
        #lr = misc.imread(self.filelist[idx])
        #lr = imageio.imread(self.filelist[idx])
        filename = self.files[idx]
        #filename="4air_"+str(int(idx_name/24))+"_"+str(idx_name%24) # 121.npy -> 4air_5_1.npy

        # if not self.train:
        #     lr=self.norm_LR_co[idx,:,:]
        #     filename="4air_"+str(idx)+".png"
        #     lr = common.set_channel([lr], self.args["n_channels"])[0]
        #     return common.np2Tensor([lr], self.args["rgb_range"])[0], -1, filename
        # else:
        # LR
        # load numpy file
        lr =np.frombuffer(self.storage.read(
                os.path.join(self.dataroot_L,filename),
                decode = None), dtype='float32')
        # 1D -> 2D
        lr = lr[32:].reshape(*self.lr_shape)

        if self.args['mostra_canali']:
            #from matplotlib import pyplot as plt
            #plt.imshow(lr)
            np.save('tmp/mostra_canali/'+filename, lr)


        if not self.args['inference'] :
            # cut the borders
            lr = lr[:-1, :-1]
            # HR
            hr =self.storage.read(
                    os.path.join(self.dataroot_H,filename),
                    decode = None)
            # 1D -> 2D
            hr = hr.reshape(*self.hr_shape)
            # cut the borders
            hr = hr[:-1,:-1]
            #hr=np.load(self.HR_path[idx])[:-1,:-1]
            hr = common.set_channel([hr], self.args["n_channels"])[0]
            data={
                "L":common.np2Tensor([lr], self.args["rgb_range"])[0],"H":common.np2Tensor([hr], self.args["rgb_range"])[0],
                "filename": filename
                 }
        else:
            # 1D -> 2D
            lr = lr.reshape(*self.lr_shape)
            # cut the borders
            lr = lr[:-1, :-1]
            # lr=np.load(self.LR_path[idx])[:-1,:-1]
            lr = common.set_channel([lr], self.args["n_channels"])[0]
            data = {
                "L": common.np2Tensor([lr], self.args["rgb_range"])[0],
                "filename": filename
            }
        return data
            #return common.np2Tensor([lr], self.args["rgb_range"])[0], common.np2Tensor([hr], self.args["rgb_range"])[0], idx, self.args["scale"]
    
    def __len__(self):
        return len(self.LR_path)
    
    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale


