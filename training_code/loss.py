import numpy as np
import torch
import torch.nn as nn


def get_loss(result, net_gt, net_mask, pred_img, loss_type):
    mask = torch.eq(torch.sum(net_gt, dim=1, keepdim=True), 0.0)
    mask = -mask.type(torch.float32) + 1.0

    if loss_type == 'mae':
        return mae_loss(result, net_gt, mask)
    elif loss_type == 'AL':
        return angular_loss(result,net_gt, mask)
    elif loss_type == 'TAL':
        return truncated_angular_loss(result,net_gt,mask)
    # elif loss_type == 'mae_and_mse':
    #     return mae_loss(result, net_gt, mask) + mse_loss(pred_img, img_gt, mask)
    # elif loss_type == 'mse':
    #     return mse_loss(pred_img, img_gt, mask)

def mse_loss(img1, img2, mask):
    mse = nn.MSELoss()
    output = mse(img1, img2)
    output = nn.functional.l1_loss(img1, img2)
    # output += nn.functional.l1_loss(img1[:,0,:,:]-img1[:,2,:,:], img2[:,0,:,:]-img2[:,2,:,:])
    # output += nn.functional.l1_loss(img1[:,1,:,:]-img1[:,3,:,:], img2[:,1,:,:]-img2[:,3,:,:])
    return output

def mae_loss(img1, img2, mask):
    mae_map = -torch.sum(img1 * img2, dim=1, keepdims=True) + 1
    loss_map = torch.abs(mae_map * mask)
    loss = torch.mean(loss_map)
    return loss

def angular_loss(img1,img2,mask):
    # img1 B*3*H*W 
    prediction_error = torch.cosine_similarity(img1, img2, dim=1, eps=1e-6)
    prediction_error = prediction_error[:,None,:,:]

    acos_mask = mask.float() \
            * (prediction_error.detach() < 0.999).float() * (prediction_error.detach() > -0.999).float()
    acos_mask = acos_mask > 0.0

 
    angular_loss = torch.mean(torch.acos(prediction_error[acos_mask]))

    return angular_loss

def truncated_angular_loss(img1, img2, mask):
    prediction_error = torch.cosine_similarity(img1, img2, dim=1, eps=1e-6)
    prediction_error = prediction_error[:,None,:,:]
    # Robust acos loss
    acos_mask = mask.float() \
            * (prediction_error.detach() < 0.9999).float() * (prediction_error.detach() > 0.0).float()
    cos_mask = mask.float() * (prediction_error.detach() <= 0.0).float()
    acos_mask = acos_mask > 0.0
    cos_mask = cos_mask > 0.0
    truncated_angular_loss = torch.sum(torch.acos(prediction_error*acos_mask)) - torch.sum(prediction_error*cos_mask)
    truncated_angular_loss = truncated_angular_loss / (torch.sum(cos_mask) + torch.sum(acos_mask))
    return truncated_angular_loss

def anglar_within_angle(mae_map_valid, angle=11.25):
    return float(mae_map_valid[mae_map_valid < angle].shape[0]) / float(mae_map_valid.shape[0])

def get_mae(net_gt, pred_camera_normal, net_mask):
 
    h, w = pred_camera_normal.shape[-2:]

    net_gt = net_gt[:, :, :h,:w]
    # print(pred_camera_normal.shape, net_gt.shape)
    pred_camera_normal = pred_camera_normal[:, :, :h,:w]

    net_mask = net_mask[:, :, :h,:w]

    valid_mask = net_mask *  torch.where(torch.mean(net_gt, dim=1, keepdim=True) != 0 , torch.ones(torch.mean(net_gt, dim=1, keepdim=True).size()).cuda(), torch.zeros(torch.mean(net_gt, dim=1, keepdim=True).size()).cuda())
    mae_map = torch.sum(net_gt * pred_camera_normal, dim=1, keepdim=True)#.clip(-1,1)
    mae_map = torch.acos(mae_map) * 180. / np.pi

    b = net_gt.size(0)
    maes = []
    ae11s = []
    ae22s = []
    ae30s = []
    for i in range(b):
        mae_map_i = mae_map[i:i+1]
        valid_mask_i = torch.mean(valid_mask[i:i+1], dim=1)
        mae_map_valid = mae_map_i[(valid_mask_i * mae_map_i)>0]
        mae = torch.mean(mae_map_valid)
        if float(mae_map_valid.shape[0])>0:
            ae11 = anglar_within_angle(mae_map_valid, 11.25)
            ae22 = anglar_within_angle(mae_map_valid, 22.5)
            ae30 = anglar_within_angle(mae_map_valid, 30)
        else:
            ae11 = 0.0
            ae22 = 0.0
            ae30 = 0.0
        maes.append(mae.item())
        ae11s.append(ae11)
        ae22s.append(ae22)
        ae30s.append(ae30)
    
    return maes, ae11s, ae22s, ae30s
