# @GonzaloMartinGarcia
# This file houses our dataset mixer and training dataset classes.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import random
import pandas as pd
import cv2
import argparse
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from skimage import io, transform
import numpy as np
import glob
import tqdm
from PIL import Image
import torch
from imgaug import augmenters as iaa
import pandas as pd
from torch.utils.data import DataLoader

    
def read_img(filename):
    img = np.array(Image.open(filename).convert('RGB'))
    return img


def read_synthesis_depth_png(file_path, mask_threshold=50000):
    """
    Reads a 16-bit grayscale PNG depth image and converts it to a normalized floating-point depth map.
    
    Args:
        file_path (str): Path to the 16-bit PNG depth image.
    
    Returns:
        depth_norm (np.ndarray): Depth values normalized to the [0, 1] range.
    """
    # Read the image with unchanged flag to preserve 16-bit depth
    depth_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"Failed to load image at {file_path}")
    
    # Verify that the image is indeed 16-bit
    if depth_image.dtype != np.uint16:
        raise ValueError("Image is not 16-bit")
    
    # get mask from depth, max value is invalid depth, set it to 0
    mask = depth_image <= mask_threshold

    # Convert the 16-bit image to floating-point and normalize to [0, 1]
    depth_norm = depth_image.astype(np.float32) / 65535.0
    
    return depth_norm, mask

def read_synthesis_depth_8bit_png(file_path, mask_threshold=128):
    """
    Reads a 16-bit grayscale PNG depth image and converts it to a normalized floating-point depth map.
    
    Args:
        file_path (str): Path to the 16-bit PNG depth image.
    
    Returns:
        depth_norm (np.ndarray): Depth values normalized to the [0, 1] range.
    """
    # Read the image with unchanged flag to preserve 16-bit depth
    depth_image =cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

    if depth_image is None:
        raise ValueError(f"Failed to load image at {file_path}")
    
    # Verify that the image is indeed 16-bit
    if depth_image.dtype != np.uint16:
        raise ValueError("Image is not 16-bit")
    
    # get mask from depth, max value is valid depth, set it to 1
    mask = depth_image <= mask_threshold

    # Convert the 16-bit image to floating-point and normalize to [0, 1]
    depth_norm = depth_image.astype(np.float32) / 255.0
    
    return depth_norm, mask

def read_synthesis_normal_png(file_path):
    # Open and convert the image to RGB
    normal_image = Image.open(file_path).convert('RGB').resize((512, 512), resample=Image.NEAREST)

    # Convert to NumPy array and normalize to [0, 1]
    normal_array = np.array(normal_image).astype(np.float32) / 255

    # Split into R, G, B channels
    r = normal_array[:, :, 0]
    g = normal_array[:, :, 1]
    b = normal_array[:, :, 2]

    # Decode to [-1, 1] range
    px = r * 2.0 - 1.0
    py = g * 2.0 - 1.0
    pz = b * 2.0 - 1.0

    # Compute magnitude and normalize
    magnitude = np.sqrt(px**2 + py**2 + pz**2) + 1e-10  # Avoid division by zero
    nx = px / magnitude
    ny = py / magnitude
    nz = pz / magnitude

    # Stack normalized components
    normalized_normal_map = np.stack((nx, ny, nz), axis=2)
    normalized_normal_map *= -1
    return normalized_normal_map

def read_photoface_dataset(depth_path, normal_path, mask_path):
    normal_map = np.load(normal_path)
    normal_map = change_axis_coordinate(normal_map)

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_norm = depth.astype(np.float32) / 65535.0
    # read mask as grayscale image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 255 is for valid pixel, 0 is for invalid pixel
    mask = mask == 255
    depth_norm[~mask] = 10.
    normal_map[~mask] = np.array([0., 0., -1.])
    return depth_norm, normal_map, mask

    
def change_axis_coordinate(normal):
    tt = np.zeros_like(normal)
    tt[:, :, 0] = normal[:, :, 1]
    tt[:, :, 1] = normal[:, :, 0]
    tt[:, :, 2] = -normal[:, :, 2]
    return tt

def read_depth_normal_synthesis(depth_path, normal_path):
    depth, mask = read_synthesis_depth_png(depth_path)
    depth[~mask] = 10.
    normal = read_synthesis_normal_png(normal_path)
    normal[~mask] = np.array([0., 0., -1.])
    return depth, normal, mask


class SynthesisDataset(Dataset):
    def __init__(self, data_dir, csv_path, dataset_name = "synthesis",
                 transform=None):
        super(SynthesisDataset, self).__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.img_size = (512, 512)
        self.csv_path = csv_path
        self.dataset_name = dataset_name
        self.samples = []

        # read data path from csv file with 3 headers (dataset, rgb, depth, normal)
        self.data_info = pd.read_csv(csv_path, header=None)
        self.num_img = len(self.data_info)
        print("Number of train images: ", self.num_img)


        self.samples = []
        for image_idx in range(self.num_img):
            sample = dict()
            sample['rgb'] = os.path.join(self.data_dir, self.data_info.iloc[image_idx, 0])
            sample['depth'] = os.path.join(self.data_dir, self.data_info.iloc[image_idx, 1])
            sample['normal'] = os.path.join(self.data_dir, self.data_info.iloc[image_idx, 2])
            if self.dataset_name == "photoface":
                sample['mask'] = os.path.join(self.data_dir, self.data_info.iloc[image_idx, 3])

            # data augmentation args
            sample['RandomHorizontalFlip'] = 0.4
            sample['distortion_prob'] = 0.05
            sample['to_gray_prob'] = 0.1

            self.samples.append(sample)


    def __getitem__(self, index):
        sample = {}
        sample['domain'] = torch.Tensor([1., 0., 0.]) # indoor
        H, W = self.img_size
        try:
            sample_path = self.samples[index]
            sample['rgb'] = read_img(sample_path['rgb'])  # [H, W, 3]
        except Exception as e:
            print("Error at index: ", sample_path['rgb'])
            sample_path = self.samples[index + 1]
            sample['rgb'] = read_img(sample_path['rgb'])
            
        if self.dataset_name == "photoface":
            sample['depth'], sample['normal'], sample["mask"] = read_photoface_dataset(sample_path['depth'], sample_path['normal'], sample_path['mask'])
        else:
            sample['depth'], sample['normal'], sample["mask"] = read_depth_normal_synthesis(sample_path['depth'], sample_path['normal'])

        H_ori, W_ori = sample['rgb'].shape[:2]

        # 1. Random Crop
        if H_ori >= H and W_ori >= W:
            H_start, W_start = np.random.randint(0, H_ori-H+1), np.random.randint(0, W_ori-W+1)
            sample['rgb'] = sample['rgb'][H_start:H_start + H, W_start:W_start + W]
            sample['depth'] = sample['depth'][H_start:H_start + H, W_start:W_start + W]
            sample['normal'] = sample['normal'][H_start:H_start + H, W_start:W_start + W]

        # 2. Random Horizontal Flip
        if np.random.random() < sample_path['RandomHorizontalFlip']:
            sample['rgb'] = np.copy(np.fliplr(sample['rgb']))
            sample['depth'] = np.copy(np.fliplr(sample['depth']))
            sample['normal'] = np.copy(np.fliplr(sample['normal']))
            sample['normal'][:,:,0] *= -1.

        # 3. Photometric Distortion
        to_gray_prob = sample_path['to_gray_prob']
        distortion_prob = sample_path['distortion_prob']
        brightness_beta = np.random.uniform(-32, 32)
        contrast_alpha = np.random.uniform(0.5, 1.5)
        saturate_alpha = np.random.uniform(0.5, 1.5)
        rand_hue = np.random.randint(-18, 18)

        brightness_do = np.random.random() < distortion_prob
        contrast_do = np.random.random() < distortion_prob
        saturate_do = np.random.random() < distortion_prob
        rand_hue_do = np.random.random() < distortion_prob

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = 0 if np.random.random() > 0.5 else 1
        if np.random.random() < to_gray_prob:
            sample['rgb'] = iaa.Grayscale(alpha=(0.8, 1.0))(image=sample['rgb'])
        else:
            # random brightness
            if brightness_do:
                alpha, beta = 1.0, brightness_beta
                sample['rgb'] = np.clip((sample['rgb'].astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)

            if mode == 0:
                if contrast_do:
                    alpha, beta = contrast_alpha, 0.0
                    sample['rgb'] = np.clip((sample['rgb'].astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)

            # random saturation
            if saturate_do:
                img = cv2.cvtColor(sample['rgb'][:,:,::-1], cv2.COLOR_BGR2HSV)
                alpha, beta = saturate_alpha, 0.0
                img[:,:,1] = np.clip((img[:,:,1].astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)
                sample['rgb'] = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)[:,:,::-1]

            # random hue
            if rand_hue_do:
                img = cv2.cvtColor(sample['rgb'][:,:,::-1], cv2.COLOR_BGR2HSV)
                img[:, :, 0] = (img[:, :, 0].astype(int) + rand_hue) % 180
                sample['rgb'] = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)[:,:,::-1]

            # random contrast
            if mode == 1:
                if contrast_do:
                    alpha, beta = contrast_alpha, 0.0
                    sample['rgb'] = np.clip((sample['rgb'].astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)

        # 4. To Tensor
        sample['rgb'] = (torch.from_numpy(np.transpose(sample['rgb'].copy(), (2, 0, 1))) / 255.) * 2.0 - 1.0  # [3, H, W]
        sample['depth'] = torch.from_numpy(sample['depth'][None].copy())  # [1, H, W]
        sample['normal'] = torch.from_numpy(np.transpose(sample['normal'].copy(), (2, 0, 1)))  # [3, H, W]

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    
# Get Dataset Here
def prepare_dataset(data_dir=None,
                    csv_path=None,
                    dataset_name='synthesis',
                    batch_size=1,
                    test_batch=1,
                    datathread=4,
                    logger=None):

    # set the config parameters
    dataset_config_dict = dict()
    
    train_dataset = SynthesisDataset(data_dir=data_dir, csv_path=csv_path, dataset_name=dataset_name)

    img_height, img_width = train_dataset.get_img_size()

    datathread = datathread
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))
    
    if logger is not None:
        logger.info("Use %d processes to load data..." % datathread)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)
    
    num_batches_per_epoch = len(train_loader)
    
    dataset_config_dict['num_batches_per_epoch'] = num_batches_per_epoch
    dataset_config_dict['img_size'] = (img_height,img_width)
    
    return train_loader, dataset_config_dict


def resize_max_res_tensor(input_tensor, mode, recom_resolution=512):
    # assert input_tensor.shape[1]==3
    original_H, original_W = input_tensor.shape[2:]
    downscale_factor = min(recom_resolution/original_H, recom_resolution/original_W)
    
    if mode == 'normal':
        resized_input_tensor = F.interpolate(input_tensor,
                                            scale_factor=downscale_factor,
                                            mode='nearest')
    else:
        resized_input_tensor = F.interpolate(input_tensor,
                                            scale_factor=downscale_factor,
                                            mode='bilinear',
                                            align_corners=False)
    
    return resized_input_tensor

def depth_scale_shift_normalization(depth):

    bsz = depth.shape[0]

    depth_ = depth[:,0,:,:].reshape(bsz,-1).cpu().numpy()
    min_value = torch.from_numpy(np.percentile(a=depth_,q=2,axis=1)).to(depth)[...,None,None,None]
    max_value = torch.from_numpy(np.percentile(a=depth_,q=98,axis=1)).to(depth)[...,None,None,None]

    normalized_depth = ((depth - min_value)/(max_value-min_value+1e-5) - 0.5) * 2
    normalized_depth = torch.clip(normalized_depth, -1., 1.)

    return normalized_depth