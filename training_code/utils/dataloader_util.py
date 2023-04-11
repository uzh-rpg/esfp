import numpy as np
import torch


def ToTensor(sample_list):
    """Convert ndarrays in sample to Tensors."""

    def SampleToTensor(sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # image = image.transpose((2, 0, 1))
        sample_tensor = torch.from_numpy(sample.copy().astype(np.float32)).permute(2,0,1)
        
        return sample_tensor
    
    return [SampleToTensor(sample) for sample in sample_list]
