#%%
from .unet import UNet
import torch.nn as nn
import torch
import numpy as np

def get_model(args):
    if not hasattr(args, "channel"):
        channel = 32
    else:
       channel =  args.channels
    if args.model=="unet":
        model = UNet(channel, 3, dim=args.dim, norm=args.norm)
    return model
#%%
if __name__ == "__main__":
    class Dict2Class(object):
        def __init__(self, my_dict):
            
            for key in my_dict:
                setattr(self, key, my_dict[key])
    
