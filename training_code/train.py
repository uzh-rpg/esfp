
# Main function for training
import argparse
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import random
import time
import warnings

import cv2
import models as Models
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from loss import get_loss, get_mae
from torch.utils.data import DataLoader
from utils.log_util import (AverageMeter, ProgressMeter, count_parameters,
                            init_logging, resume_training,
                            save_model_checkpoint)
from utils.train_util import (adjust_learning_rate, get_net_input_channel,
                              get_net_input_cuda)
from utils.visualizer import (markdown_visualizer_img, save_image, tensor2im,
                              tensor2norm)

from dataloader import create_dataloader

parser = argparse.ArgumentParser(description="Train the normal estimation network")
# Convenient Training mode
parser.add_argument("--training_mode", type=str, default="all", 
                        choices=["all", "test"],	
                        help='quick training mode switch, fast or all, or validation.\
                        this argument overwrites the following arguments')

parser.add_argument("--exp_name", type=str, default="none", \
                        help='Name of this experiments')

# Model parameters
parser.add_argument("--model", type=str, default='unet', choices=["unet"],
                help="model_to_use")

parser.add_argument("--dim", type=int, default=64,
                help="the dim of feature")
parser.add_argument("--residual_num", type=int, default=8, 
                help="the number of residual blocks in u-net")
parser.add_argument("--norm", type=str, default='in',
                help="normalization method")


# Dataset parameters
parser.add_argument('--dataset', default='mitsuba', type=str, 
                    choices= ["mitsuba", "realevents", "realimages", "realim_motion", "realev_motion"], 
                    help='path to dataset') 

# Data Preprocessing parameters
parser.add_argument("--batch_size", type=int, default=1, 	\
                help="Training batch size")

parser.add_argument("--debug", action='store_true',\
                    help="use debug mode")

parser.add_argument('--netinput', default='events_event_intensities', type=str, 
                choices= [  
                            "fourraw",
                            "fourraw_pol",
                            "events_8_bins_voxel_grid",
                            "events_8_bins_cvgri",
                            ], 
                help='feature feed into the netowrk') 

# Training parameters
parser.add_argument("--epochs", "--e", type=int, default=1000, \
                    help="Number of total training epochs")

parser.add_argument("--cos", action='store_true',\
                    help="use cos decay learning rate")

parser.add_argument("--lr", type=float, default=1e-4, \
                    help="Initial learning rate")
parser.add_argument("--dropout", type=float, default=0., \
                    help="Dropout rate")
parser.add_argument("--save_freq", type=int, default=50,\
                    help="Number of training epochs to save state")
parser.add_argument("--print_freq", type=int, default=10,\
                    help="Number of training epochs to save state")
parser.add_argument("--loss", type=str, default="mae", 
                    choices = ["mae","mse", "mae_and_mse"],
                    help='loss to train the network')
parser.add_argument('--pretrained', default=None, type=str, help='path to pretrained network') 

# Dirs
parser.add_argument('--dataroot', default='/tmp/cluster/mitsuba_dataset/event_data/new_dataset/', type=str, help='path to dataset') 

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')



def main():
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)

def evaluate_normals(pred, gt):
    valid_mask = np.where(np.mean(gt, axis=2) != 0 , 1, 0)
    mae_map = np.sum(gt * pred, axis=2).clip(-1,1)
    mae_map = np.arccos(mae_map) * 180. / np.pi

    mae_map_gray = np.uint8((mae_map*5*255.0*valid_mask/180).clip(0, 255))
    diff_color = cv2.applyColorMap(mae_map_gray, cv2.COLORMAP_JET)
    eps=1e-5

    angle = 11.25
    ae_11 = float(mae_map_valid[mae_map_valid < angle].shape[0]) / float(mae_map_valid.shape[0]+eps)
    angle = 22.5
    ae_22 = float(mae_map_valid[mae_map_valid < angle].shape[0]) / float(mae_map_valid.shape[0]+eps)
    angle = 30
    ae_30 = float(mae_map_valid[mae_map_valid < angle].shape[0]) / float(mae_map_valid.shape[0]+eps)
    
    return (np.mean(mae_map_valid), 
            diff_color,
            np.median(mae_map_valid), 
            np.sqrt(((mae_map_valid) ** 2).mean()), 
            ae_11, 
            ae_22, 
            ae_30,
            mae_map*valid_mask, 
            valid_mask)

def validate(loader_val, model, args, epoch=None, writer=None):
    batch_time = AverageMeter('Data+Forward Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mae_score = AverageMeter('MAE', ':6.2f')
    ae11_metric = AverageMeter('ae11_metric', ':6.2f')
    ae22_metric = AverageMeter('ae22_metric', ':6.2f')
    ae30_metric = AverageMeter('ae30_metric', ':6.2f')
    progress = ProgressMeter(
        len(loader_val),
        [batch_time, losses, mae_score, ae11_metric, ae22_metric, ae30_metric],
        prefix='Test: ')
    # switch to evaluate mode
    save_image_dir = "{}/validation".format(args.output_dir) if args.training_mode == "validation" else "{}/{}".format(args.output_dir, epoch) 
    os.makedirs(save_image_dir, exist_ok=True)
    model.eval()
    mae_list = []

    result_txt = os.path.join(save_image_dir, 'results.txt')
    metrics = ["mean_ae", "med_ae", "rmse_ae", "ae_11", "ae_22", "ae_30", "fillrate"]
    metrics_dict = {}
    
    for metric in metrics:
        metrics_dict[metric] = []
    with torch.no_grad():
        end = time.time()
        print(len(loader_val))
        for i, data_sample in enumerate(loader_val, 0):
            # end = time.time()
            # data_sample = [item.cuda(args.gpu) for item in data_sample] if args.distributed else [item.cuda() for item in data_sample]
            f_idx = data_sample[1]
            data_sample = [item.cuda() for item in data_sample[0]]
            net_mask, net_image, net_pol, net_gt = data_sample

            net_in = get_net_input_cuda(net_image, net_pol, args)
            # preprocessing the data using gpu
            
            result = model(net_in)
            pred_normal = result[0]
            pred_img = result[1]
            pred_camera_normal = F.normalize(pred_normal, p=2, dim=1)
            loss = get_loss(pred_camera_normal, net_gt, net_mask, pred_img, args.loss)
            mae, ae11, ae22, ae30 = get_mae(net_gt, pred_camera_normal, net_mask)
            ae11_metric_item = np.mean(ae11)
            ae22_metric_item = np.mean(ae22)
            ae30_metric_item = np.mean(ae30)
            losses.update(loss.item(), net_in.size(0))
            # losses.update(np.mean(mae), net_in.size(0))
            mae_score.update(np.mean(mae), net_in.size(0))
            mae_list.extend(mae)
            ae11_metric.update(ae11_metric_item, net_in.size(0))
            ae22_metric.update(ae22_metric_item, net_in.size(0))
            ae30_metric.update(ae30_metric_item, net_in.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            b = pred_camera_normal.size(0)
            for img_idx in range(b):
                pred_camera_normal[img_idx:img_idx+1][net_mask==0]=0
                # pred_img[img_idx:img_idx+1][torch.eq(torch.sum(net_mask, dim=1, keepdim=True))==0]=0
                # import pdb; pdb.set_trace()
                save_image(tensor2norm(pred_camera_normal[img_idx:img_idx+1]), "{}/im{}_pred.jpg".format(save_image_dir, i * b + img_idx))
                save_image(tensor2im(net_mask), "{}/im{}_pred_img.jpg".format(save_image_dir, i * b + img_idx))
                # if net_in.shape[1] == 1:
                    # gray_input = torch.rep net_in[img_idx:img_idx+1,:3]
                gray_input = net_in[img_idx:img_idx+1, 0].repeat(1,3,1,1)
                # elif net_in.shape[1]>4:
                #     gray_input = torch.sum(net_in[img_idx:img_idx+1],dim=1).repeat(1,3,1,1)
                # elif net_in.shape[1] == 2:
                #     gray_input = abs(net_in[img_idx:img_idx+1, 0].repeat(1,3,1,1))
                # elif net_in.shape[1]==4:
                #     gray_input = net_in[img_idx:img_idx+1,:3]#.repeat(1,3,1,1)
                save_image(tensor2im(gray_input, cent=0, factor=255.), "{}/im{}_in.jpg".format(save_image_dir, i * b + img_idx))
                save_image(tensor2norm(net_gt[img_idx:img_idx+1]), "{}/im{}_gt.jpg".format(save_image_dir, i * b + img_idx))
                # save_image(tensor2im(net_img_gt[img_idx:img_idx+1,:3]), "{}/im{}_gt_img.jpg".format(save_image_dir, i * b + img_idx))
                mean_ae = mae[img_idx]
                ae_11= ae11[img_idx]
                ae_22= ae22[img_idx]
                ae_30= ae30[img_idx]
                with open(result_txt, "a") as f:
                    for idx, metric in enumerate([mean_ae, 0, 0, ae_11, ae_22, ae_30, 0]):
                        f.writelines("{} {} {}\n".format(f_idx[img_idx], metrics[idx], metric))
                        metrics_dict[metrics[idx]].append(metric)
                f.close()


            if i % args.print_freq == 0:
                progress.display(i)
        markdown_visualizer_img(save_image_dir, i * b + img_idx, mae_list)
        print(' * MAE {mae_score.avg:.3f} ae11_metric {ae11_metric.avg:.3f}, ae22_metric {ae22_metric.avg:.3f}, ae30_metric {ae30_metric.avg:.3f}'
              .format(mae_score=mae_score, ae11_metric=ae11_metric, ae22_metric=ae22_metric, ae30_metric=ae30_metric))
        
    with open(result_txt, "a") as f:
        for metric in metrics:
            print("{} {}: {}\n".format("average", metric, np.mean(metrics_dict[metric])))
            f.writelines("{} {}: {:.3f}\n".format("average", metric, np.mean(metrics_dict[metric])))
    f.close()
    if writer is not None:
        print("writer test loss: ", losses.avg, epoch)
        writer.add_scalar('test loss', losses.avg, epoch)
        writer.add_scalar('test mae', mae_score.avg, epoch)
    return losses.avg, mae_score.avg

def trainsave(loader_train, epoch, model, args):
    # switch to evaluate mode
    mae_score = AverageMeter('MAE', ':6.2f')
    ae11_metric = AverageMeter('ae11_metric', ':6.2f')
    ae22_metric = AverageMeter('ae22_metric', ':6.2f')
    ae30_metric = AverageMeter('ae30_metric', ':6.2f')
    save_image_dir = "{}/{}".format(args.output_dir_train, epoch) 
    os.makedirs(save_image_dir, exist_ok=True)
    model.eval()
    mae_list = []
    result_txt = os.path.join(save_image_dir, 'results.txt')
    metrics = ["mean_ae", "ae_11", "ae_22", "ae_30"]
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric] = []

    with torch.no_grad():
        for i, data_sample in enumerate(loader_train, 0):
            f_idx = data_sample[1]
            data_sample = [item.cuda() for item in data_sample[0]]
            net_mask, net_image, net_pol, net_gt  = data_sample

            net_in = get_net_input_cuda(net_image, net_pol, args)
            # preprocessing the data using gpu
            
            result = model(net_in)
            pred_normal = result[0]
            pred_img = result[1]
            pred_camera_normal = F.normalize(pred_normal, p=2, dim=1)
            mae, ae11, ae22, ae30 = get_mae(net_gt, pred_camera_normal, net_mask)
            ae11_metric_item = np.mean(ae11)
            ae22_metric_item = np.mean(ae22)
            ae30_metric_item = np.mean(ae30)
            mae_score.update(np.mean(mae), net_in.size(0))

            mae_list.extend(mae)
            ae11_metric.update(ae11_metric_item, net_in.size(0))
            ae22_metric.update(ae22_metric_item, net_in.size(0))
            ae30_metric.update(ae30_metric_item, net_in.size(0))


            b = pred_camera_normal.size(0)
            for img_idx in range(b):
                pred_camera_normal[img_idx:img_idx+1][net_mask[img_idx:img_idx+1]==0]=0
                # pred_img[img_idx:img_idx+1][torch.eq(torch.sum(net_mask, dim=1, keepdim=True))==0]=0
                # pred_img[img_idx:img_idx+1][net_mask==0]=0
                save_image(tensor2im(pred_camera_normal[img_idx:img_idx+1]), "{}/im{}_pred.jpg".format(save_image_dir, i * b + img_idx))
                save_image(tensor2im(pred_img[img_idx:img_idx+1,:3]), "{}/im{}_pred_img.jpg".format(save_image_dir, i * b + img_idx))
                if net_in.shape[1] == 8 or net_in.shape[1] == 4:
                    gray_input = torch.sum(abs(net_in[img_idx:img_idx+1,:,:,:]), dim=1).repeat(1,3,1,1)
                else:
                    gray_input = net_in[img_idx:img_idx+1,:3]
                save_image(tensor2im(gray_input, cent=0, factor=255.), "{}/im{}_in.jpg".format(save_image_dir, i * b + img_idx), rs=True)
                save_image(tensor2im(net_gt[img_idx:img_idx+1]), "{}/im{}_gt.jpg".format(save_image_dir, i * b + img_idx))
                # save_image(tensor2im(net_img_gt[img_idx:img_idx+1,:3]), "{}/im{}_gt_img.jpg".format(save_image_dir, i * b + img_idx))

                mean_ae = mae[img_idx]
                ae_11= ae11[img_idx]
                ae_22= ae22[img_idx]
                ae_30= ae30[img_idx]
                with open(result_txt, "a") as f:
                    for idx, metric in enumerate([mean_ae, ae_11, ae_22, ae_30]):
                        f.writelines("{} {} {}\n".format(f_idx[img_idx], metrics[idx], metric))
                        metrics_dict[metrics[idx]].append(metric)
                f.close()
            markdown_visualizer_img(save_image_dir, i * b + img_idx, mae_list)
            print(' * MAE {mae_score.avg:.3f} ae11_metric {ae11_metric.avg:.3f}, ae22_metric {ae22_metric.avg:.3f}, ae30_metric {ae30_metric.avg:.3f}'
                .format(mae_score=mae_score, ae11_metric=ae11_metric, ae22_metric=ae22_metric, ae30_metric=ae30_metric))
            if i==5:
                break
        with open(result_txt, "a") as f:
                for metric in metrics:
                    print("{} {}: {}\n".format("average", metric, np.mean(metrics_dict[metric])))
                    f.writelines("{} {}: {:.3f}\n".format("average", metric, np.mean(metrics_dict[metric])))
        f.close()
        return

def train_one_epoch(loader_train, epoch, model, optimizer, writer, args):
    batch_time = AverageMeter('Forward Time', ':6.3f')
    data_time = AverageMeter('Data Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(loader_train),
        [batch_time, data_time, losses],
    prefix="Epoch: [{}]".format(epoch))        
    data_time_start = time.time()
    step = 0
    model.train()
    for i, data_sample in enumerate(loader_train, 0):
        data_time.update(time.time() - data_time_start)
        batch_time_start = time.time()
        optimizer.zero_grad()
        # data_sample = [item.cuda(args.gpu) for item in data_sample] if args.distributed else [item.cuda() for item in data_sample]
        # f_idx = data_sample[1]
        data_sample =  [item.cuda() for item in data_sample[0]]
        # preprocessing the data using gpu
        net_mask, net_image, net_pol, net_gt = data_sample
        net_in = get_net_input_cuda(net_image, net_pol, args)
        if i==0:
            print('net_in.shape ',net_in.shape)
        result = model(net_in)
        pred_normal = result[0]
        pred_img = result[1]
        pred_camera_normal = F.normalize(pred_normal, p=2, dim=1)

        loss = get_loss(pred_camera_normal, net_gt, net_mask, pred_img, args.loss)
        mae, ae11, ae22, ae30 = get_mae(net_gt, pred_camera_normal, net_mask)
        # loss = torch.mean(torch.as_tensor(mae))

        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - batch_time_start)
        losses.update(loss.item(), net_in.size(0))
        if step % args.print_freq == 0:
            progress.display(i)
        step += 1
        data_time_start = time.time()
        if args.debug:
            break        
    


    if writer is not None:
        # tensorboard logger
        writer.add_scalar('training loss', losses.avg, epoch)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
    return model, losses.avg

#%%
def main_worker(gpu, ngpus_per_node, args):
    r"""Performs the main training loop
    """
    # Optimizer

    args.gpu = gpu
    print(args.gpu)


    args.log_dir = "./logs/"+ args.exp_name
    args.output_dir = "./results/"+ args.exp_name 
    args.output_dir_train = "./results/"+ args.exp_name +'/train'
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir_train, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ckpt_epochs'), exist_ok=True)
    
    args.use_mitsuba = True if "mitsuba" in args.dataset else False
    args.use_realevents = True if "realevents" in args.dataset else False
    args.use_realimages = True if "realimages" in args.dataset else False
    args.use_realimages_motion = True if "realim_motion" in args.dataset else False
    args.use_realevents_motion = True if "realev_motion" in args.dataset else False
    args.split = "inter"

    def print_info():
        print("\n### Training shape from polarization model ###")
        print("> Parameters:")
        for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
            print('\t{}: {}'.format(p, v))
        print('\n')
    print_info()    



    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # creating models
    writer = None
    writer, logger = init_logging(args)
    
    args.channel = get_net_input_channel(args)
    print("args.channel", args.channel)

    
    model = Models.get_model(args)

    print(model)
    print("args.model", args.model)

    

    if not torch.cuda.is_available():
        args.gpu = None
        print('using CPU, this will be slow')
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    start_epoch = resume_training(args, model)
    cudnn.benchmark = True


    num_params = count_parameters(model)
    print("num_params", num_params)
    print("args.training_mode == %s"%args.training_mode)
   
    
    sfp_train_dataset, sfp_test_dataset = create_dataloader(args)

    train_sampler = None
    if args.training_mode=='all':
        loader_train = DataLoader(
            sfp_train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
        loader_val = DataLoader(sfp_test_dataset, 
                                batch_size =1, 
                                shuffle=False, num_workers=8, 
                                drop_last=False)
        print("\t# of training samples: %d\n" % int( len(loader_train)))



        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print('learning rate %f' % args.lr)
        
        # start from last epoch to avoid overwritting the previous result
        best_mae = 100
        for epoch in range(start_epoch, args.epochs):
            if args.cos:
                adjust_learning_rate(optimizer, epoch, args)
            model, _ = train_one_epoch(loader_train, epoch, model, optimizer, writer, args)
            if (epoch+1) % args.save_freq == 0 or epoch==0:
                save_model_checkpoint(model, epoch, save_path=os.path.join(args.output_dir, 'ckpt_epochs/ckpt_e{}.pth'.format(epoch)))    
                save_model_checkpoint(model, epoch, save_path=os.path.join(args.output_dir, 'ckpt.pth'))    
                trainsave(loader_train, epoch, model, args)
                _, mean_mae = validate(loader_val, model, args, epoch, writer)
                if mean_mae < best_mae:
                    best_mae = mean_mae
                    save_model_checkpoint(model, epoch, save_path=os.path.join(args.output_dir, 'ckpt_best_val.pth'))    
    elif args.training_mode=='test' :               
        loader_val = DataLoader(sfp_test_dataset, 
                                batch_size =1, 
                                shuffle=False, num_workers=8, 
                                drop_last=False)
        print("\t# of testing samples: %d\n" % int( len(loader_val)))
        epoch = 0
        _, mean_mae = validate(loader_val, model, args, epoch, writer)

        
 
if __name__ == "__main__":
    main()    
