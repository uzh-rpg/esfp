# MIT License
# Copyright (c) 2022 Chenyang LEI
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import hsv_to_rgb

class polarization:
    # Fundamental property
    I_0 = None
    I_45 = None
    I_90 = None
    I_135 = None
    # Function
    def __init__(self,I_0,I_45,I_90,I_135):
        self.I_0 = I_0
        self.I_45 = I_45
        self.I_90 = I_90
        self.I_135 = I_135
        I = (self.I_0 + self.I_45 + self.I_90 + self.I_135) / 2.
        Q = self.I_0 - self.I_90 
        U = self.I_45 - self.I_135
        Iun = I * 0.5
        self.Iun = Iun
        Q[Q == 0] = 1e-10
        I[I == 0] = 1e-10
        self.rho = (np.sqrt(np.square(Q)+np.square(U))/I).clip(0,1)
        phi = 0.5 * np.arctan(U/Q)
        cos_2phi = np.cos(2*phi)
        check_sign = cos_2phi * Q
        phi[check_sign<0] =  phi[check_sign<0] + math.pi/2.

    def visualize_phi_hsv(self):
        phi = self.phi / math.pi
        
        h,w = phi.shape[:2]
        hsv = np.concatenate([phi[...,None], np.ones([h,w,1]),np.ones([h,w,1])],axis=2)
        self.phi_rgb = hsv_to_rgb(hsv)
        return  self.phi_rgb

    def visualize_rho_hsv(self):
        h,w = self.rho.shape[:2]
        rho = -0.67 * (self.rho - 1.0)
        hsv = np.concatenate([rho[..., None], np.ones([h,w,1]),np.ones([h,w,1])],axis=2)
        self.rho_rgb = hsv_to_rgb(hsv)
        return  self.rho_rgb
    
    def visualize_polarimg(self):
        h,w = self.rho.shape[:2]
        hsv = np.concatenate([phi[...,None]/math.pi, rho[...,None], np.ones([h,w,1])],axis=2)
        self.polarimg_rgb = hsv_to_rgb(hsv)
        return  self.polarimg_rgb


def visualize_phi_hsv(phi):
    h,w = phi.shape[:2]
    hsv = np.concatenate([phi[...,None], np.ones([h,w,1]),np.ones([h,w,1])],axis=2)
    phi_rgb = hsv_to_rgb(hsv)
    return  phi_rgb


def prepare_shadow_mask(all_Iun, light_num=4):
    all_Iun = all_Iun[..., :light_num]

    # Light enough
    mask_light = all_Iun.copy()
    mask_light[mask_light < 0.0003] = 0
    mask_light[mask_light > 0.0003] = 1
    
    # Not minimum
    mask_not_min = all_Iun.copy()
    all_Iun_min = np.min(all_Iun, axis=3)[..., None]
    mask_not_min = mask_not_min - all_Iun_min
    mask_not_min[mask_not_min>0] = 1.

    
    mask = np.max(np.concatenate([mask_light[..., None], mask_not_min[..., None]],axis=4),axis=4)
    
    mask = np.concatenate([mask, np.ones_like(mask[...,:1])], axis=3)
    # print("mask", mask.shape)
    print(np.min(np.mean(mask[...,:4], axis=3)), np.max(np.mean(mask[...,:4], axis=3)),np.mean(mask[...,:4]))
    return mask


def save_and_concat_imgs(output_path='result.jpg', mask=None, PolarRaw=None, rho_mask=None, Iun=None, normal=None,
                         normal2=None, albedo=None, eta=None, zenith=None, azimuth=None, phi=None, rho=None,
                         lights=None):
    for i in [PolarRaw, normal, normal2, rho_mask, mask]:
        print(i.shape)
    out_img = np.concatenate([PolarRaw, normal * mask, normal2 * mask, rho_mask], axis=1).clip(0, 1)
    out_img = out_img[..., ::-1]
    if lights.shape[0] < 5:
        cv2.imwrite(output_path, np.uint8(out_img * 255.0))
    else:
        out_light = np.concatenate([lights[..., 3 * i:3 * i + 3] for i in range(lights.shape[2] // 3)], axis=1)
        out_light = (out_light + 1.) / 2.
        out_all = np.concatenate([out_img, out_light], axis=0)
        cv2.imwrite(output_path, np.uint8(out_all * 255.0))

    # Calculates Rotation Matrix given euler angles.


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R



def soft_rho_mask(all_rho, cropped_mask):
    nonzero_values = all_rho[np.nonzero(cropped_mask)]  # nonzero?
    sorted_nonzero_values = np.sort(nonzero_values, axis=None)  # flatten
    threshold_percentile = 20  # (drop 20% pixels, use 70%)
    low_threshold = sorted_nonzero_values[int(len(sorted_nonzero_values) * threshold_percentile / 100.0)]
    low_threshold = np.maximum(low_threshold, 0.02)
    high_threshold = sorted_nonzero_values[int(len(sorted_nonzero_values) * 0.9)]
    high_threshold = np.minimum(high_threshold, 0.9)
    # mask1 soft mask
    high_pass_rho_mask = all_rho / (all_rho + low_threshold) * (1 + low_threshold)
    low_pass_rho_mask = (1 - all_rho) / (2 - all_rho - high_threshold) * (2 - high_threshold)
    mid_pass = high_pass_rho_mask * low_pass_rho_mask
    rho_mask = mid_pass / np.mean(mid_pass[np.nonzero(cropped_mask)])
    # print('low_threshold %.4f, high_threshold %.4f' % (low_threshold, high_threshold))
    return rho_mask



def prepare_outdoor_data(img_name, light_num=4, uv=False, use_mask=True):
    print(img_name)
    ambient = cv2.imread(img_name, -1) / 65535.
    mask = np.ones([1024,1224,1])*1.0
    h_start, h_end, w_start, w_end = mask_to_crop(mask)
    cropped_mask = mask[h_start:h_end, w_start:w_end, :]

    all_imgs = []
    for i in range(5):
        all_imgs.append(prepare_data_from_raw(ambient, input_type="img"))
    # print(all_imgs[i][0].shape, all_imgs[i][1].shape, all_imgs[i][2].shape, all_imgs[i][3].shape)
    all_raw_imgs = np.concatenate([all_imgs[i][0] for i in range(5)], axis=-1)[h_start:h_end, w_start:w_end, :]\
                   * cropped_mask
    all_Iun = np.concatenate([all_imgs[i][1] for i in range(5)], axis=-1)[h_start:h_end, w_start:w_end, :]\
              * cropped_mask
    all_phi = np.concatenate([all_imgs[i][2] for i in range(5)], axis=-1)[h_start:h_end, w_start:w_end, :] \
              * cropped_mask
    all_rho = np.concatenate([all_imgs[i][3] for i in range(5)], axis=-1)[h_start:h_end, w_start:w_end, :] \
              * cropped_mask

    h,w = 2048, 2448
    u = np.tile(np.arange(w), [h, 1])
    v = np.tile(np.arange(h)[..., None], [1, w])
    uv1 = np.concatenate([u[..., None], v[..., None], 1. * np.ones([h, w, 1])], axis=2)[::2, ::2]
    all_uv1 = uv1[None, ...]
    return all_raw_imgs[np.newaxis, ...], all_Iun[np.newaxis, ...], all_phi[np.newaxis, ...], all_rho[np.newaxis, ...], \
       cropped_mask[np.newaxis, ...], all_uv1[:, h_start:h_end, w_start:w_end, :]


def prepare_pesudo_GT_for_model(folder="../data/synthetic_3_28/perfect/", light_num=4, uv=False, use_mask=True):
    """
    Prepare Iun, phi, rho from raw and mask. We crop a small rectangle area to reduce the size of model
    :param folder: path to input file, which should contain mask, raw_i(i=0,1,2,3, ...light num)
    :param light_num: number of light directions
    :return: in light_num = 5 case, raw(1, 256, 256, 20), Iun(1, 256, 256, 5), phi(1, 256, 256, 5), rho(1, 256, 256, 5),
            mask(1, 256, 256, 1)
    """
    print(folder)
    if os.path.isfile(folder + 'crop_mask.jpg'):
        mask = cv2.imread(folder + 'crop_mask.jpg', 0)[:, :, np.newaxis]
        mask = mask / float(np.max(mask))
    elif os.path.isfile(folder + 'mask.png'):
        mask = cv2.imread(folder + 'mask.png', 0)[:, :, np.newaxis]
        mask = mask / float(np.max(mask))
    else:
        mask = np.ones([1024,1224,1])*1.0#.astype("float32")
    h_start, h_end, w_start, w_end = mask_to_crop(mask)
    cropped_mask = mask[None, h_start:h_end, w_start:w_end, :]

    all_imgs = []
    ambient_path = folder + 'ambient.png'
    ambient = cv2.imread(ambient_path, -1) / 65535.
    all_imgs.append(prepare_pol_from_raw(ambient, input_type="img")[None, h_start:h_end, w_start:w_end, :])
    h,w = 2048, 2448
    u = np.tile(np.arange(w) ,[h,1])
    v = np.tile(np.arange(h)[...,None], [1,w])
    uv1 = np.concatenate([u[..., None],v[..., None], 1. * np.ones([h,w,1])],axis=2) [::2,::2]
    all_uv1 = uv1[None, h_start:h_end, w_start:w_end, :]


    normal_path = ambient_path.replace('ambient.png', 'normal.jpg')
    normal_confidence_path = ambient_path.replace('ambient.png', 'normal_confidence_mask.jpg')
    if os.path.isfile(normal_path):
        normal = (imread(normal_path)[None, ...] / 255. * 2) - 1
        normal_confidence = imread(normal_confidence_path)[None,...,None] / 255.
    else:
        normal = None
        normal_confidence = None
    return all_imgs[0], cropped_mask, all_uv1, normal, normal_confidence



def prepare_data_for_model(folder="../data/synthetic_3_28/perfect/", light_num=4, uv=False, use_mask=True):
    """
    Prepare Iun, phi, rho from raw and mask. We crop a small rectangle area to reduce the size of model
    :param folder: path to input file, which should contain mask, raw_i(i=0,1,2,3, ...light num)
    :param light_num: number of light directions
    :return: in light_num = 5 case, raw(1, 256, 256, 20), Iun(1, 256, 256, 5), phi(1, 256, 256, 5), rho(1, 256, 256, 5),
            mask(1, 256, 256, 1)
    """
    print(folder)
    if os.path.isfile(folder + 'mask.png'):
        mask = cv2.imread(folder + 'mask.png', 0)[:, :, np.newaxis]
        mask = mask / float(np.max(mask))
    else:
        mask = np.ones([1024,1224,1])*1.0#.astype("float32")
    h_start, h_end, w_start, w_end = mask_to_crop(mask)
    cropped_mask = mask[h_start:h_end, w_start:w_end, :]

    all_imgs = []
    ambient = cv2.imread(folder + 'ambient.png', -1) / 65535.

    for i in range(light_num):
        if os.path.isfile(folder + 'raw_1.png'):
            raw_png_name = folder + 'raw_{i:d}.png'.format(i=i)
            all_imgs.append(prepare_data_from_raw(raw_png_name))
        else:
            raw_png_name = folder + 'flash_{i:d}.png'.format(i=i+1)
            print(raw_png_name)
            raw_png = cv2.imread(raw_png_name, -1) / 65535.
            flashonly = (raw_png - ambient).clip(0, 1)
            all_imgs.append(prepare_data_from_raw(flashonly, input_type="img"))

    all_imgs.append(prepare_data_from_raw(folder + 'ambient.png'))
    all_raw_imgs = np.concatenate([all_imgs[i][0] for i in range(5)], axis=-1)[h_start:h_end, w_start:w_end, :]\
                   * cropped_mask
    all_Iun = np.concatenate([all_imgs[i][1] for i in range(5)], axis=-1)[h_start:h_end, w_start:w_end, :]\
              * cropped_mask
    all_phi = np.concatenate([all_imgs[i][2] for i in range(5)], axis=-1)[h_start:h_end, w_start:w_end, :] \
              * cropped_mask
    all_rho = np.concatenate([all_imgs[i][3] for i in range(5)], axis=-1)[h_start:h_end, w_start:w_end, :] \
              * cropped_mask

    h,w = 2048, 2448
    u = np.tile(np.arange(w),[h,1])
    v = np.tile(np.arange(h)[...,None],[1,w])
    uv1 = np.concatenate([u[...,None],v[...,None],1.*np.ones([h,w,1])],axis=2)[::2,::2]
    all_uv1 = uv1[None,...]
    return all_raw_imgs[np.newaxis, ...], all_Iun[np.newaxis, ...], all_phi[np.newaxis, ...], all_rho[np.newaxis, ...], \
       cropped_mask[np.newaxis, ...], all_uv1[:, h_start:h_end, w_start:w_end, :]

def prepare_pols_for_model(folder="../data/synthetic_3_28/perfect/", light_num=4, uv=False, use_mask=True):
    """
    Prepare Iun, phi, rho from raw and mask. We crop a small rectangle area to reduce the size of model
    :param folder: path to input file, which should contain mask, raw_i(i=0,1,2,3, ...light num)
    :param light_num: number of light directions
    :return: in light_num = 5 case, raw(1, 256, 256, 20), Iun(1, 256, 256, 5), phi(1, 256, 256, 5), rho(1, 256, 256, 5),
            mask(1, 256, 256, 1)
    """
    print(folder)
    if os.path.isfile(folder + 'mask.png') and use_mask:
        mask = cv2.imread(folder + 'mask.png', 0)[:, :, np.newaxis]
        mask = mask / float(np.max(mask))
    else:
        mask = np.ones([1024,1224,1])*1.0#.astype("float32")
    h_start, h_end, w_start, w_end = mask_to_crop(mask)
    cropped_mask = mask[None, h_start:h_end, w_start:w_end, :]

    all_imgs = []
    ambient = cv2.imread(folder + 'ambient.png', -1) / 65535.
    for i in range(light_num):
        if os.path.isfile(folder + 'raw_1.png'):
            raw_png_name = folder + 'raw_{i:d}.png'.format(i=i)
            all_imgs.append(prepare_pol_from_raw(raw_png_name)[None, h_start:h_end, w_start:w_end, :])
        else:
            print(folder + 'flash_{i:d}.png'.format(i=i+1))
            raw_png = cv2.imread(folder + 'flash_{i:d}.png'.format(i=i+1), -1) / 65535.
            flashonly = (raw_png - ambient).clip(0, 1)
            all_imgs.append(prepare_pol_from_raw(flashonly, input_type="img")[None, h_start:h_end, w_start:w_end, :])

    all_imgs.append(prepare_pol_from_raw(ambient, input_type="img")[None, h_start:h_end, w_start:w_end, :])

    h,w = 2048, 2448
    u = np.tile(np.arange(w) ,[h,1])
    v = np.tile(np.arange(h)[...,None], [1,w])
    uv1 = np.concatenate([u[..., None],v[..., None], 1. * np.ones([h,w,1])],axis=2) [::2,::2]
    all_uv1 = uv1[None, h_start:h_end, w_start:w_end, :]

    return (all_imgs[0], all_imgs[1], all_imgs[2], all_imgs[3], all_imgs[4], cropped_mask, all_uv1)

def prepare_pols_withOrgFlash_for_model(folder="../data/synthetic_3_28/perfect/", light_num=4, uv=False, use_mask=True):
    """
    Prepare Iun, phi, rho from raw and mask. We crop a small rectangle area to reduce the size of model
    :param folder: path to input file, which should contain mask, raw_i(i=0,1,2,3, ...light num)
    :param light_num: number of light directions
    :return: in light_num = 5 case, raw(1, 256, 256, 20), Iun(1, 256, 256, 5), phi(1, 256, 256, 5), rho(1, 256, 256, 5),
            mask(1, 256, 256, 1)
    """
    print(folder)
    if os.path.isfile(folder + 'mask.png') and use_mask:
        mask = cv2.imread(folder + 'mask.png', 0)[:, :, np.newaxis]
        mask = mask / float(np.max(mask))
    else:
        mask = np.ones([1024,1224,1])*1.0#.astype("float32")
    h_start, h_end, w_start, w_end = mask_to_crop(mask)
    cropped_mask = mask[None, h_start:h_end, w_start:w_end, :]

    all_imgs = []
    ambient = cv2.imread(folder + 'ambient.png', -1) / 65535.
    for i in range(light_num):
        if os.path.isfile(folder + 'raw_1.png'):
            flashonly_name = folder + 'raw_{i:d}.png'.format(i=i)
            flashonly = cv2.imread(flashonly_name, -1) / 65535.
            flash = (flashonly + ambient).clip(0, 1)
        else:
            print(folder + 'flash_{i:d}.png'.format(i=i+1))
            flash = cv2.imread(folder + 'flash_{i:d}.png'.format(i=i+1), -1) / 65535.
            flashonly = (flash - ambient).clip(0, 1)

        all_imgs.append(prepare_pol_from_raw(flashonly, input_type="img")[None, h_start:h_end, w_start:w_end, :])
        all_imgs.append(prepare_pol_from_raw(flash, input_type="img")[None, h_start:h_end, w_start:w_end, :])
    all_imgs.append(prepare_pol_from_raw(ambient, input_type="img")[None, h_start:h_end, w_start:w_end, :])

    h,w = 2048, 2448
    u = np.tile(np.arange(w) ,[h,1])
    v = np.tile(np.arange(h)[...,None], [1,w])
    uv1 = np.concatenate([u[..., None],v[..., None], 1. * np.ones([h,w,1])],axis=2) [::2,::2]
    all_uv1 = uv1[None, h_start:h_end, w_start:w_end, :]
    return (all_imgs, cropped_mask, all_uv1)

def rawimg_demosaic(raw_img):
###########################################################
#   demosaic
###########################################################
#To check
    I_90=raw_img[::2,::2] 
    I_45=raw_img[::2,1::2] 
    I_135=raw_img[1::2,::2] 
    I_0=raw_img[1::2,1::2]
    return I_0, I_45, I_90, I_135

def prepare_pol_from_raw(raw_png_name, input_type="name"):
    '''
    raw png to [raw, Iun, phi, rho]
    :param raw_png_name: .png file saved by cv2.imwrite(uint16)
    :return:  raw + Iun + phi + rho (h, w, 7)
    '''
    if input_type == "name":
        raw_png = cv2.imread(raw_png_name, -1) / 65535.
    else:
        raw_png = raw_png_name
    I_0, I_45, I_90, I_135 = rawimg_demosaic(raw_png)
    polarImg = polarization(I_0, I_45, I_90, I_135)
    raw_imgs = np.concatenate([i[:, :, np.newaxis] for i in [I_0, I_45, I_90, I_135]], axis=2)
    pol_imgs = np.concatenate([i[:, :, np.newaxis] for i in [polarImg.Iun, polarImg.phi, polarImg.rho]], axis=2)
    return np.concatenate([raw_imgs, pol_imgs], axis = -1)


def prepare_data_from_raw(raw_png_name, input_type="name"):
    '''
    raw png to [raw, Iun, phi, rho]
    :param raw_png_name: .png file saved by cv2.imwrite(uint16)
    :return:  raw(h, w, 4), Iun(h, w, 1), phi(h, w, 1), rho(h, w, 1)
    '''
    print(raw_png_name)
    if input_type == "name":
        raw_png = cv2.imread(raw_png_name, -1) / 65535.
    else:
        raw_png = raw_png_name
    I_0, I_45, I_90, I_135 = rawimg_demosaic(raw_png)
    polarImg = polarization(I_0, I_45, I_90, I_135)
    raw_imgs = np.concatenate([i[:, :, np.newaxis] for i in [I_0, I_45, I_90, I_135]], axis=2)
    pol_imgs = np.concatenate([i[:, :, np.newaxis] for i in [polarImg.Iun, polarImg.phi, polarImg.rho]], axis=2)
    return raw_imgs, pol_imgs[:, :, 0:1], pol_imgs[:, :, 1:2], pol_imgs[:, :, 2:3]


def mask_to_crop(mask):
    '''
    We crop a small rectangle area to reduce the size of model
    :param mask: hw1
    :return: h_start, h_end, w_start, w_end
    '''
    h, w = mask.shape[:2]
    (nonzero_h, nonzero_w, _) = np.nonzero(mask)
    h_start = np.min(nonzero_h)
    h_end = np.max(nonzero_h)
    w_start = np.min(nonzero_w)
    w_end = np.max(nonzero_w)

    h_offset = (h_end - h_start) // 32 * 32 + 32  # the bottom of the image is blocked by desk
    w_offset = (w_end - w_start) // 32 * 32 + 32

    h_end = h_start + h_offset if (h_start + h_offset <= h) else h_start + h_offset - 32
    w_end = w_start + w_offset if (w_start + w_offset <= w) else w_start + w_offset - 32
    return h_start, h_end, w_start, w_end


def phi2azimuth(phi, reflection_type="diffuse"):
    ###########################################################
    #   convert two angles to unit surface normal
    ###########################################################
    if reflection_type == "diffuse":
        azimuth1, azimuth2 = phi, phi + math.pi
    else:
        azimuth1, azimuth2 = phi-math.pi/2., phi + math.pi/2.
    return azimuth1, azimuth2 


def dop2zenith(dop, eta = 1.6, reflection_type="diffuse"):
    ###########################################################
    #   convert two angles to unit surface normal
    ###########################################################
    if reflection_type == "diffuse":
        num = (eta**4)*(1-dop**2) + (2*eta**2)*(2*dop**2+dop-1) + (dop**2 + 2*dop) - (4*eta**3)*dop*np.sqrt(1-dop**2) + 1
        num = np.clip(num,0,np.inf)
        den = (eta**4+1)*((dop+1)**2) + (2*eta**2)*(3*dop**2 + 2*dop - 1)
        cos_zenith = np.sqrt(num/den)
        # print(num.min(),num.max(),den.min(),den.max(),cos_zenith.min(),cos_zenith.max())
        zenith = np.arccos(cos_zenith)
        zeniths = [zenith]
    else:
        split = 1000
        dop = np.uint16(dop*split)
    #         print(dop.shape)
        lookup_table1,lookup_table2 = Create_specular_lookuptable(split=split, eta = eta)
        zenith1 = np.vectorize(lookup_table1.get)(dop)
        zenith2 = np.vectorize(lookup_table2.get)(dop)
    #         print(zenith1.max(), zenith1.min(),zenith2.max(), zenith2.min())
    #         print(zenith1.shape, zenith2.shape)
        zeniths = [zenith1,zenith2]
    return zeniths


def Create_specular_lookuptable(split= 500, eta=1.5):
    zenith2dop_dict1 = {}
    zenith2dop_dict2 = {}
    old_dop = 0
    for i in range(split):
        zenith = i * (math.pi / 2 / split)
        dop = zenith2dop(zenith)
        new_dop = int(round(dop*split))
        if dop > old_dop:
            zenith2dop_dict1[new_dop] = zenith
        else:
            zenith2dop_dict2[new_dop] = zenith
        old_dop = dop
    cnt = 0
    for i in range(split):
        if i not in zenith2dop_dict1:
    #       print("Dict1", i)
            cnt += 1
            zenith2dop_dict1[i] = zenith2dop_dict1[i-1] 
        if i not in zenith2dop_dict2:
    #        print("Dict2", i)    
            cnt += 1
            zenith2dop_dict2[i] = zenith2dop_dict2[i-1]
    #         print(i, zenith2dop_dict1[i])
    #     print(cnt)
    return  zenith2dop_dict1, zenith2dop_dict2


def zenith2dop(zenith, eta = 1.5):
###########################################################
#   For specular reflection
###########################################################
    #zenith = zenith / 2000.
    num = 2*(np.sin(zenith)**2)*np.cos(zenith)*np.sqrt(eta**2 - np.sin(zenith)**2)
    den = eta**2 - np.sin(zenith)**2 - eta**2 * (np.sin(zenith)**2) + 2* (np.sin(zenith)**4)
    dop = num / den
    return dop


def angles2normal(zenith, azimuth):
###########################################################
#   convert two angles to unit surface normal
###########################################################
    normal = np.zeros((np.shape(zenith) + (3,)))
    normal[..., 0] = np.sin(zenith) * np.cos(azimuth)
    normal[..., 1] = np.sin(zenith) * np.sin(azimuth)
    normal[..., 2] = np.cos(zenith)
    return normal


def ShapeFromPolarizationInfor(I, phi, dop, eta = 1.5, output_path="normal_from_depth", reflection_types = ["specular","diffuse"]):

    # reflection_types = ["diffuse", "specular"]
    normal_solutions = []
    cnt = 0
    for reflection_type in reflection_types:
        azimuths = phi2azimuth(phi,reflection_type=reflection_type) # 2 for diffuse or specular
        zeniths = dop2zenith(dop,eta=eta,reflection_type=reflection_type) # 1 for diffuse, 2 for specular
        for azimuth in azimuths:
            for zenith in zeniths:
                cnt += 1
                #print(cnt, reflection_type)
                #print(cnt,zenith.min(),zenith.max())
                # TODO: use too much cpu
                tmp_normal = angles2normal(zenith, azimuth)
                normal_solutions.append(tmp_normal)
                tmp_normal[np.isnan(tmp_normal)] = 0
                # plt.imshow((tmp_normal+1)/2.)
                # plt.show()
                # mask=(tmp_normal[:,:,2]<=0)
                # if (np.sum(mask)>0):
                    # print('##############:',output_path,cnt)
                # print("{}_{}.jpg".format(output_path,cnt))

                # tmp_normal=tmp_normal[...,::-1]
                # cv2.imwrite("{}_{}.jpg".format(output_path,cnt),np.uint8(255.*(tmp_normal+1)/2.))
                # cv2.imwrite("{}_{}_mask.jpg".format(output_path,cnt),np.uint8(mask*255.))
    return normal_solutions


def calculate_phi_rho(I_0, I_45, I_90, I_135):
    I = (I_0 + I_45 + I_90 + I_135)/ 2.
    Q = I_0 - I_90 
    U = I_45 - I_135
    Q[Q == 0] = 1e-6
    I[I == 0] = 1e-6
    rho = np.sqrt(np.square(Q)+np.square(U))/I
    phi = 0.5*np.arctan(U/Q)
    rho[rho>1] = 1 
    # print(rho.max(),rho.min())
    # Make Phi in [0, pi]
    phi = (phi + math.pi)%math.pi
    # sign of cos(2phi) = sign of Q
    cos_2phi = np.cos(2*phi)
    check_sign = cos_2phi * Q
    phi[check_sign<0] =  phi[check_sign<0] + math.pi/2.
    # print(check_sign[check_sign<0].shape, check_sign[check_sign>0].shape)
    cos_2phi = np.cos(2*phi)
    check_sign = cos_2phi * Q
    phi[check_sign<0] =  phi[check_sign<0] + math.pi/2.
    # print(check_sign[check_sign<0].shape)
    # print(check_sign[check_sign>0].shape)
    # print(phi[phi<0].shape, phi.max())
    phi = (phi + math.pi)%math.pi
        
    return 0.5*I, phi, rho

def get_concat_normals(iun, aop, dop):
    normal_solutions = ShapeFromPolarizationInfor(iun, aop, dop)
    # TODO: use too much cpu
    
    single_normal = np.concatenate([normal_solutions[0][None,...], normal_solutions[1][None,...], normal_solutions[2][None,...], 
        normal_solutions[3][None,...], normal_solutions[4][None,...], normal_solutions[5][None,...]],axis=3)
    return single_normal


def test():
    path = '../../data/iccv2021/ready_kinect3_lucid_pair/20201221/CYT_1_clean_car/set_0000_polar.npy'
    net_input = np.load(path)
    I_0     = net_input[..., 0:1]
    I_45    = net_input[..., 1:2]
    I_90    = net_input[..., 2:3]
    I_135   = net_input[..., 3:4]
    Iun     = net_input[..., 4]
    phi     = net_input[..., 5]
    rho     = net_input[..., 6]

    priors = get_concat_normals(Iun, phi, rho)
#%%
if __name__ == '__main__':
    path = '../../data/iccv2021/ready_kinect3_lucid_pair/20201221/CYT_1_clean_car/set_0000_polar.npy'
    net_input = np.load(path)
    # I_0     = net_input[..., 0:1]
    # I_45    = net_input[..., 1:2]
    # I_90    = net_input[..., 2:3]
    # I_135   = net_input[..., 3:4]
    Iun     = net_input[..., 4]
    phi     = net_input[..., 5]
    rho     = net_input[..., 6]

    priors = get_concat_normals(Iun, phi, rho)
# %%
