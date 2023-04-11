#%%
import math
import os
import random
from time import time

import cv2
import h5py
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import utils.dataloader_util as dataloader_util
from skimage import io
from torch.utils.data.dataset import Dataset

seed = 2020
np.random.seed(seed)
random.seed(seed)

class MitsubaDataset(Dataset):
    def __init__(self, 
                 dataroot, 
                 train = 1,
                 netinput = "fourraw",
                 with_viewing_dir = False
                 ):
        super().__init__()
        self.w=self.h=512
        self.dataroot = dataroot
        self.train = train
        self.with_viewing_dir = with_viewing_dir
        if self.train:
            self.split_path = os.path.join(self.dataroot, 'train')
        else:
            self.split_path = os.path.join(self.dataroot, 'test')
        self.objects_dir = [f.path for f in os.scandir(self.split_path) if f.is_dir()]
        self.length = len(self.objects_dir)

        self.use_events = False
        self.use_images = False
        self.num_images = 0
        self.num_pol=3
        self.downsample=False
        if "fourraw" in netinput:
            self.use_images=True
            self.num_images=4
            self.downsample=False
        elif "event" in netinput:
            self.use_events = True
            if "4" in netinput:
                self.num_images+=4
            if "8" in netinput:
                self.num_images+=8
            if "voxel" in netinput:
                self.event_repr = "voxel_grid"
            elif "cvgri" in netinput:
                self.event_repr = "cvgri"
    
    def __len__(self):
        return self.length

    def get_i_un_phi_rho(self, frames, num_imgs=None):
        if num_imgs==None:
            num_imgs = self.num_images
        if num_imgs==4:
            I = np.sum(frames[:,:,:4], axis=2) / 2.
            Iun = I * 0.5
            Q = frames[:,:,0] - frames[:,:,2]
            U = frames[:,:,1] - frames[:,:,3]
            Q[Q == 0] = 1e-10
            I[I == 0] = 1e-10
            # rho = np.zeros((frames.shape[0], frames.shape[1], 1))
            # phi = np.zeros((frames.shape[0], frames.shape[1], 1))

            rho = (np.sqrt(np.square(Q)+np.square(U))/I).clip(0,1)
            phi = 0.5 * np.arctan(U/Q)

        cos_2phi = np.cos(2*phi)
        check_sign = cos_2phi * Q
        phi[check_sign<0] =  phi[check_sign<0] + math.pi/2.
        phi = (phi + math.pi)%math.pi
        return Iun, rho, phi

    def __getitem__(self, index):
        '''
        '''
        object_folder = self.objects_dir[index]

        h5f = h5py.File(str(object_folder+'/events.h5'), 'r')
        gt_normals = np.array(h5f['gt_normals']) # [H, W, C]

        H, W = gt_normals.shape[:2]
        C = self.num_images
        frames = np.zeros((H, W, C))
        pol = np.zeros((H, W, self.num_pol))
        eps = 1e-5
        contrast_threshold = 0.05
        if self.use_events:
            h5f = h5py.File(str(object_folder+'/events.h5'), 'r')
            gt_normals = np.array(h5f['gt_normals']) # [H, W, C]
            if self.event_repr == "cvgri":
                frame = io.imread(os.path.join(object_folder, 'images/{:03d}.jpg'.format(0)), as_gray=True).astype(np.float32)
                event_grid = np.load(str(object_folder+'/events_vgrid_8bins.npy'))

                event_grid = np.cumsum(event_grid*contrast_threshold,axis=2)
                event_grid += np.expand_dims(frame,axis=-1) # img offset on or off
                frames = event_grid
            
            elif self.event_repr=="voxel_grid":
                frame = io.imread(os.path.join(object_folder, 'images/{:03d}.jpg'.format(0)), as_gray=True).astype(np.float32)
                voxel_grid = np.load(str(object_folder+'/events_vgrid_8bins.npy'))
                frames = voxel_grid

            else:
                print("Wrong event representation")
                voxel_grid = 0

        if self.use_images:
            thetas = [0,  45, 90,  135]
            img_idx =1 
            for i in range(len(thetas)):
                frame = io.imread(os.path.join(object_folder, 'images/{:03d}.jpg'.format(thetas[i])), as_gray=True).astype(np.float32)
                if len(frame.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
                    frame = np.expand_dims(frame, -1)
                frames[:,:,i]=frame[:,:,0]
                img_idx+=1
            i_un, phi, rho = self.get_i_un_phi_rho(frames, num_imgs=4)
            pol[:,:,0] = i_un
            pol[:,:,1] = phi
            pol[:,:,2] = rho
        
        
        frames[~np.isfinite(frames)] = 0
        pol[~np.isfinite(pol)] = 0
        # frames = (frames- np.expand_dims(np.min(frames, axis=-1), axis=-1))/(np.expand_dims(np.max(frames, axis=-1), axis=-1)-np.expand_dims(np.min(frames, axis=-1), axis=-1) +eps)
        net_in = frames
        net_pol = pol
        net_gt = gt_normals
        net_mask = np.ones([self.h, self.w, 3])
        net_mask[np.mean(net_gt, axis=2) == 0] = [0,0,0]

        sample_list = net_mask, net_in, net_pol, net_gt
        tensor_list = dataloader_util.ToTensor(sample_list)
        return tensor_list, object_folder


class RealEventMotionDatasetTest(Dataset):
    def __init__(self,
                 dataroot, 
                 train = 0
                 ):
        super().__init__()
        self.w = 1280
        self.h= 720
        self.dataroot = dataroot
        self.train = train
        self.split_path = self.dataroot
        self.objects_files = sorted([f.path for f in os.scandir(self.split_path) if f.path.endswith('.h5')])
        self.length = len(self.objects_files)

        self.use_events = False
        self.use_images = False
        self.num_images = 4
        self.num_pol=3
        self.downsample=False

    
    def __len__(self):
        return self.length
    
    def _bil_w(self, x, x_int):
        return 1 - np.abs(x_int- x)
    
    def _draw_xy_to_voxel_grid(self, voxel_grid, x, y, b, value):
        if x.dtype == np.uint16:
            self._draw_xy_to_voxel_grid_int(voxel_grid, x, y, b, value)
            return

        x_int = x.astype("int32")
        y_int = y.astype("int32")
        for xlim in [x_int, x_int + 1]:
            for ylim in [y_int, y_int + 1]:
                weight = self._bil_w(x, xlim) * self._bil_w(y, ylim)
                self._draw_xy_to_voxel_grid_int(voxel_grid, xlim, ylim, b, weight * value)

    def _draw_xy_to_voxel_grid_int(self, voxel_grid, x, y, b, value):
        H, W, B = voxel_grid.shape
        mask = (x >= 0) & (y >= 0) & (x < W) & (y < H)
        np.add.at(voxel_grid, (y[mask], x[mask], b[mask]), value[mask])

    def events_to_voxel_grid(self, events, bins, normalize=False, t0_us=None, t1_us=None):
        """
        Build a voxel grid with trilinear interpolation in the time and x,y domain from a set of events.
        """
        voxel_grid =  np.zeros((self.h, self.w, bins), np.float32)
        if len(events) < 2:
            return voxel_grid
        
        # normalize the event timestamps so that they lie between 0 and num_bins
        t0_us = t0_us if t0_us is not None else events['t'][0]
        t1_us = t1_us if t1_us is not None else events['t'][-1]
        deltaT = t1_us - t0_us

        if deltaT == 0:
            deltaT = 1.0

        t_norm = (bins -1) * (events['t'] - t0_us) / deltaT

        t_norm_int = t_norm.astype("int32")
        for tlim in [t_norm_int, t_norm_int+1]:
            mask = (tlim >= 0) & (tlim < bins)
            weight = self._bil_w(t_norm_int, tlim) * events['p']
            self._draw_xy_to_voxel_grid(voxel_grid, events['x'][mask], events['y'][mask], tlim[mask], weight[mask])

        return voxel_grid

    def __getitem__(self, index):
        '''
        '''
        object_file = self.objects_files[index]
        print(object_file)
        h5f = h5py.File(str(object_file), 'r')
        events = dict()
        events['x'] = h5f['x']
        events['y'] = h5f['y']
        events['t'] = h5f['t']
        events['p'] = h5f['p']
        gt_normals = np.ones((self.h, self.w, 3))
        # gt_normals = np.load(str(object_folder+'/normal_gt2.npy')).astype(np.float32) # [H, W, C]
        H, W = gt_normals.shape[:2]
        C = self.num_images
        frames = np.zeros((H, W, C+1))
        eps = 1e-5
        contrast_threshold = 0.27
        # if os.path.isfile(str(object_file.split('.h5')[0]+'_vgrid_4bins.npy')):
        #    voxel_grid = np.load(str(object_file.split('.h5')[0]+'_vgrid_4bins.npy')) # [H, W, C]
        # else:
        voxel_grid = self.events_to_voxel_grid(events, bins=4)
        # np.save(str(object_file.split('.h5')[0]+'_vgrid_4bins.npy'), voxel_grid)
        # voxel_grid[np.sum(voxel_grid, axis=-1)<=5]==0
        # frame = cv2.imread(os.path.join(object_folder, 'Ev_{:04d}.png'.format(0)), 0).astype(np.float32)/255
        cumsum_vgrid = np.cumsum(voxel_grid*contrast_threshold,axis=2)
        # cumsum_vgrid += np.expand_dims(frame,axis=-1)
        frames[:,:,1:] = voxel_grid# cumsum_vgrid
        net_in = frames[:,:,1:]
        net_pol =  np.zeros((H, W, self.num_pol))
        net_gt = gt_normals#((gt_normals+0.5)/255.0)-0.5
        net_img_gt =  np.zeros((H, W, 4))# np.expand_dims(img_frames, -1)
        net_mask = np.ones([self.h, self.w, 3])
        net_mask[np.mean(net_in, axis=2) == 0] = [0,0,0]
        net_coordinate = self.image_coordinate[0,:self.h,:self.w]
        viewing_direction = self.viewing_direction[0,:self.h,:self.w]

        sample_list = net_mask, net_in, net_pol, net_gt, net_img_gt, net_coordinate, viewing_direction
        tensor_list = dataloader_util.ToTensor(sample_list)
        
        tensor_list = [T.CenterCrop(512)(t) for t in tensor_list]
        return tensor_list, object_file


class RealDataset(Dataset):
    def __init__(self,
                 dataroot,
                 train=0,
                 use_events=True):
        super().__init__()
        self.w = 1280
        self.h = 720
        self.dataroot = dataroot
        self.train = train
        self.split_path = self.dataroot
        if self.train:
            self.split_path = os.path.join(self.dataroot, 'train')
            self.objects_dir = [f.path for f in os.scandir(self.split_path) if f.is_dir()]
        else:
            self.split_path = os.path.join(self.dataroot, 'test')
            self.objects_dir = [f.path for f in os.scandir(self.split_path) if f.is_dir()]
        self.length = len(self.objects_dir)

        self.use_events = use_events
        if self.use_events:
            self.num_images = 8
        else:
            self.num_images = 4
        self.num_pol = 3
        self.downsample= True
    
    def __len__(self):
        return self.length
    
    def _bil_w(self, x, x_int):
        return 1 - np.abs(x_int- x)
    
    def _draw_xy_to_voxel_grid(self, voxel_grid, x, y, b, value):
        if x.dtype == np.uint16:
            self._draw_xy_to_voxel_grid_int(voxel_grid, x, y, b, value)
            return

        x_int = x.astype("int32")
        y_int = y.astype("int32")
        for xlim in [x_int, x_int + 1]:
            for ylim in [y_int, y_int + 1]:
                weight = self._bil_w(x, xlim) * self._bil_w(y, ylim)
                self._draw_xy_to_voxel_grid_int(voxel_grid, xlim, ylim, b, weight * value)

    def _draw_xy_to_voxel_grid_int(self, voxel_grid, x, y, b, value):
        H, W, B = voxel_grid.shape
        mask = (x >= 0) & (y >= 0) & (x < W) & (y < H)
        np.add.at(voxel_grid, (y[mask], x[mask], b[mask]), value[mask])

    def events_to_voxel_grid(self, events, bins, normalize=False, t0_us=None, t1_us=None):
        """
        Build a voxel grid with trilinear interpolation in the time and x,y domain from a set of events.
        """
        voxel_grid =  np.zeros((self.h, self.w, bins), np.float32)
        if len(events) < 2:
            return voxel_grid
        
        # normalize the event timestamps so that they lie between 0 and num_bins
        t0_us = t0_us if t0_us is not None else events['t'][0]
        t1_us = t1_us if t1_us is not None else events['t'][-1]
        deltaT = t1_us - t0_us

        if deltaT == 0:
            deltaT = 1.0

        t_norm = (bins -1) * (events['t'] - t0_us) / deltaT

        t_norm_int = t_norm.astype("int32")
        for tlim in [t_norm_int, t_norm_int+1]:
            mask = (tlim >= 0) & (tlim < bins)
            weight = self._bil_w(t_norm_int, tlim) * events['p']
            self._draw_xy_to_voxel_grid(voxel_grid, events['x'][mask], events['y'][mask], tlim[mask], weight[mask])

        return voxel_grid

    def __getitem__(self, index):
        H = self.h
        W = self.w
        object_folder = self.objects_dir[index]
        C = self.num_images
        frames = np.zeros((H, W, C))
        contrast_threshold = 0.27
        if self.use_events:
            if os.path.isfile(str(object_folder+'/vgrid.npy')):
                voxel_grid = np.load(str(object_folder+'/vgrid.npy')) # [H, W, C]
            else:
                h5f = h5py.File(str(object_folder+'/events.h5'), 'r')
                events = dict()
                events['x'] = h5f['x']
                events['y'] = h5f['y']
                events['t'] = h5f['t']
                events['p'] = h5f['p']
                voxel_grid = self.events_to_voxel_grid(events, bins=8)
                np.save(str(object_folder+'/vgrid.npy'), voxel_grid)
            frame = cv2.imread(os.path.join(object_folder, 'Ev_{:04d}.png'.format(0)), 0).astype(np.float32)/255
            frames = np.cumsum(voxel_grid*contrast_threshold,axis=2)
            frames += np.expand_dims(frame,axis=-1)
        else:
            thetas = [0, 45, 90, 135]
            img_idx = 0
            for i in range(len(thetas)):
                frame = cv2.imread(os.path.join(object_folder, 'Ev_{:04d}.png'.format(thetas[i])), 0).astype(np.float32)/255
                if len(frame.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
                    frame = np.expand_dims(frame, -1)
                    frames[:,:,i]=frame[:,:,0]
        gt_normals = np.load(str(object_folder+'/normal_gt2.npy')).astype(np.float32) # [H, W, C]
        
        net_in = frames
        net_pol =  np.zeros((H, W, self.num_pol))
        net_gt =   gt_normals
        net_img_gt =  np.zeros((H, W, 4))
        net_mask = np.ones([self.h, self.w, 3])
        
        sample_list = net_mask, net_in, net_pol, net_gt
        tensor_list = dataloader_util.ToTensor(sample_list)
        if self.downsample:
            tensor_list = [T.Resize(size=(360, 360))(T.CenterCrop(size=(720))(t)) for t in tensor_list]
        else:
            tensor_list = [T.CenterCrop(size=(720))(t) for t in tensor_list]
        return tensor_list, object_folder


def create_dataloader(args):
    print('> Loading datasets ...')
    if args.use_mitsuba:
        sfp_train_dataset   = MitsubaDataset(args.dataroot,
                                    netinput = args.netinput,                                
                                    train=1)
        sfp_test_dataset    = MitsubaDataset(args.dataroot, 
                                    netinput = args.netinput,                             
                                    train=0)
    elif args.use_realevents:
        sfp_train_dataset = RealDataset(args.dataroot, train=1, use_events=True)
        sfp_test_dataset = RealDataset(args.dataroot, train=0, use_events=True)

    return sfp_train_dataset, sfp_test_dataset
