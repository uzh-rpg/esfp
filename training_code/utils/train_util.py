import cv2
import torch
import numpy as np
import math


def get_net_input_cuda(input_images, input_pol, ARGS):
    '''
        preprocessing the input data using gpu
        add positional encoding
        WARNING: I deleted phi_rgb for convenience, no sure the performance diff
    '''

    cat_list = []
    # import pdb; pdb.set_trace()
    if "onlyiun" in ARGS.netinput:
        cat_list.append(input_pol[:,0])
    if "fourraw" in ARGS.netinput:# or "twelveraw" in ARGS.netinput:
        assert input_images.shape[1]==4
        cat_list.append(input_images[:,:4])
    if "twelveraw" in ARGS.netinput:
        assert input_images.shape[1]==12
        cat_list.append(input_images[:,:12])
    # if "eightraw" in ARGS.netinput:
    #     assert input_images.shape[1]==8
    #     cat_list.append(input_images[:,:8])
    # if "fiveraw" in ARGS.netinput:
    #     assert input_images.shape==5
    #     cat_list.append(input_images[:,:5])
    if "events" in ARGS.netinput:
        if "cvgri" in ARGS.netinput:
            # print("L32", input_images.shape)
            if "4" in ARGS.netinput:
                assert input_images.shape[1]==4
                cat_list.append(input_images)
            if "8" in ARGS.netinput:
                assert input_images.shape[1]==8
                cat_list.append(input_images)
        elif "voxel_grid" in ARGS.netinput:
            if "4" in ARGS.netinput:
                assert input_images.shape[1]==4
                cat_list.append(input_images[:,:4])
            elif "8" in ARGS.netinput:
                assert input_images.shape[1]==8
                cat_list.append(input_images[:,:8])
            elif "12" in ARGS.netinput:
                assert input_images.shape[1]==12
                cat_list.append(input_images[:,:12])
            elif "5" in ARGS.netinput:
                assert input_images.shape[1]==5
                cat_list.append(input_images[:,:5])
        elif "combined_grid" in ARGS.netinput:
            assert input_images.shape[1]==9
            cat_list.append(input_images[:,:9])
        elif "event_intensities" in ARGS.netinput:
            assert input_images.shape[1]==4
            cat_list.append(input_images[:,:4])
    if "pol" in ARGS.netinput:
        phi = input_pol[:,1:2] # ablation on input as phi information
        phi_encode = torch.cat([torch.cos(2 * phi), torch.sin(2 * phi)], axis=1)
        cat_list.append(phi_encode)
        cat_list.append(input_pol[:,2:])
        
        if "rawphi" in ARGS.netinput:
            phi_encode = phi
                        
    
    net_input = torch.cat(cat_list, axis=1)  
    return net_input

def get_net_input_channel(ARGS):
    '''
        preprocessing the input data using gpu
        add positional encoding
        WARNING: I deleted phi_rgb for convenience, no sure the performance diff
    '''

    channel = 0
    raw_img = 1
    # import pdb; pdb.set_trace()
    if "onlyiun" in ARGS.netinput:
        channel += 1
    if "fourraw" in ARGS.netinput:
        channel += 4
        # raw_img = 1
    if "twelveraw" in ARGS.netinput:
        channel += 12
        # channel += 4
        # raw_img = 3
    if "eightraw" in ARGS.netinput:
        channel += 8
        # channel += 4
        # raw_img = 2
    if "fiveraw" in ARGS.netinput:
        channel += 5
    if "events" in ARGS.netinput:
        if "cvgri" in ARGS.netinput:
            if "4" in ARGS.netinput:
                channel += 4
            elif "8" in ARGS.netinput:
                channel += 8
            elif "12" in ARGS.netinput:
                channel += 12
        elif "voxel_grid" in ARGS.netinput:
            if "4" in ARGS.netinput:
                channel += 4
            elif "8" in ARGS.netinput:
                channel += 8
            elif "12" in ARGS.netinput:
                channel += 12
            elif "5" in ARGS.netinput:
                channel += 5
        elif "combined_grid" in ARGS.netinput:
            channel += 9
        elif "event_intensities" in ARGS.netinput:
            channel += 4
    if "pol" in ARGS.netinput:
        if "rawphi" in ARGS.netinput:
            channel += 2
        else:
            channel += 3
    
    return channel


def adjust_learning_rate(optimizer, epoch, args):
    
    """Decay the learning rate based on schedule"""
    lr = args.lr
    args.warmup_epochs = 0
    if epoch < args.warmup_epochs:
        lr *=  float(epoch) / float(max(1.0, args.warmup_epochs))
        if epoch == 0 :
            lr = 1e-6
    else:
        # progress after warmup        
        if args.cos:  # cosine lr schedule
            # lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
            progress = float(epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
            lr *= 0.5 * (1. + math.cos(math.pi * progress)) 
            # print("adjust learning rate now epoch %d, all epoch %d, progress"%(epoch, args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Epoch-{}, base lr {}, optimizer.param_groups[0]['lr']".format(epoch, args.lr), optimizer.param_groups[0]['lr'])
