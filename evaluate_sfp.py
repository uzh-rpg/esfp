import numpy as np
import argparse
import glob
import os
import h5py
import math
import cv2

from polarization_utils import ShapeFromPolarizationInfor, get_concat_normals

def read_events(filename, has_gt):
    h5f = h5py.File(str(filename), 'r')
    x = h5f['x'][:]
    y = h5f['y'][:]
    t = h5f['t'][:]
    p = h5f['p'][:]
    if has_gt:
        gt_normal = np.array(h5f['gt_normals'])
    else:
        gt_normal = 0
    return x,y,t,p, gt_normal

def evaluate_normals(pred, gt):
    # valid_mask = np.ones(np.mean(pred, axis=2).shape)
    pred[~np.isfinite(pred)] = 0
    valid_mask = np.where((np.mean(gt, axis=2)+np.mean(pred, axis=2)) != 0 , 1, 0)
    valid_mask[np.mean(pred, axis=2) == 0]=0
    # print(valid_mask.shape)
    mae_map = np.sum(gt * pred, axis=2).clip(-1,1)
    mae_map = np.arccos(mae_map) * 180. / np.pi

    mae_map_gray = np.uint8((mae_map*5*255.0*valid_mask/180).clip(0, 255))
    diff_color = cv2.applyColorMap(mae_map_gray, cv2.COLORMAP_JET)

    mae_map_valid = mae_map[(valid_mask)>0]
    if mae_map_valid.shape[0]>0:
        angle = 11.25
        ae_11 = float(mae_map_valid[mae_map_valid < angle].shape[0]) / float(mae_map_valid.shape[0])
        angle = 22.5
        ae_22 = float(mae_map_valid[mae_map_valid < angle].shape[0]) / float(mae_map_valid.shape[0])
        angle = 30
        ae_30 = float(mae_map_valid[mae_map_valid < angle].shape[0]) / float(mae_map_valid.shape[0])
    else:
        ae_11 = 0
        ae_22 = 0
        ae_30 = 0
    
    return (np.mean(mae_map_valid), 
            diff_color,
            np.median(mae_map_valid), 
            np.sqrt(((mae_map_valid) ** 2).mean()), 
            ae_11, 
            ae_22, 
            ae_30,
            mae_map*valid_mask, 
            valid_mask)

def get_aop_dop_from_images(images: np.ndarray, num_images=4):
    if num_images==4:
        I_0 = images[:,:,0]
        I_45 = images[:,:,3]
        I_90 = images[:,:,6]
        I_135 = images[:,:,9]
        I = (I_0 + I_45 + I_90 + I_135)/ 2.

        Q = I_0 - I_90 
        U = I_45 - I_135
        Q[Q == 0] = 1e-6
        I[I == 0] = 1e-6
        rho = np.sqrt(np.square(Q)+np.square(U))/I
        phi =0.5*np.arctan(U/Q)

    elif num_images==8:
        i0 = images[:,:,0]
        i30 = images[:,:,2]
        i45 = images[:,:,3]
        i75 = images[:,:,5]
        i90 = images[:,:,6]
        i120 = images[:,:,8]
        i135 = images[:,:,9]
        i165 = images[:,:,11]
        I = (i0+i30+i45+i75+i90+i120+i135+i165) / 4.
        Q = Q1 = i0 - i90
        Q3 = i30 - i120
        Q1[Q1 == 0] = 1e-10
        Q3[Q3 == 0] = 1e-10

        U1 = i45 - i135
        U3 = i75 - i165
        I[I == 0] = 1e-10
        Iun = I * 0.5
        rho = (0.5*(np.sqrt(np.square(Q1)+np.square(U1)) +  np.sqrt(np.square(Q3)+np.square(U3)) )/  (2*I)).clip(0,1)
        phi = (0.5 * np.arctan(U1/Q1) + 0.5 * (np.arctan(U3/Q3)-np.deg2rad(60)))/2

    elif num_images==12:
        i0 = images[:,:,0]
        i15 = images[:,:,1]
        i30 = images[:,:,2]
        i45 = images[:,:,3]
        i60 = images[:,:,4]
        i75 = images[:,:,5]
        i90 = images[:,:,6]
        i105 = images[:,:,7]
        i120 = images[:,:,8]
        i135 = images[:,:,9]
        i150 = images[:,:,10]
        i165 = images[:,:,11]

        I = np.sum(images, axis=2) / 6.
        Q = Q1 = i0 - i90
        Q2 = i15 - i105
        Q3 = i30 - i120
        Q1[Q1 == 0] = 1e-10
        Q2[Q2 == 0] = 1e-10
        Q3[Q3 == 0] = 1e-10

        U1 = i45 - i135
        U2 = i60 - i150
        U3 = i75 - i165
        I[I == 0] = 1e-10
        Iun = I * 0.5

        rho = ((np.sqrt(np.square(Q1)+np.square(U1)) +  np.sqrt(np.square(Q3)+np.square(U3))+ np.sqrt(np.square(Q2)+np.square(U2)) )/ (3*I)).clip(0,1)
        phi =  0.5 *( np.arctan(U1/Q1) +  np.arctan(U3/Q3)-np.deg2rad(60) + np.arctan(U2/Q2)-np.deg2rad(30)) / 3.

    rho[rho>1] = 1
    phi = (phi + math.pi)%math.pi
    # sign of cos(2phi) = sign of Q
    cos_2phi = np.cos(2*phi)
    check_sign = cos_2phi * Q
    phi[check_sign<0] =  phi[check_sign<0] + math.pi/2.
    cos_2phi = np.cos(2*phi)
    check_sign = cos_2phi * Q
    phi[check_sign<0] =  phi[check_sign<0] + math.pi/2.

    phi = (phi + math.pi)%math.pi
    aop= phi
    dop = rho
    return aop, dop

def get_aop_dop_from_events(x: np.ndarray, y: np.ndarray, p: np.ndarray, t: np.ndarray, resolution: list, images:np.ndarray, object_dir: str):
    c = 0.035
    phi_img = [i for i in range(0, 180, 15)]
    aop = np.zeros((resolution[0], resolution[1]))
    dop = np.zeros((resolution[0], resolution[1]))
    mask = np.zeros((resolution[0], resolution[1]))
    load=False
    count = 0
    if load:
        event_intensities = np.load(object_dir+'/events_delta_t.npy')
    else:
        event_intensities = np.zeros((resolution[0], resolution[1], 12))
        for plot_y in np.unique(y):
            for plot_x in np.unique(x):
                xy = x[y==plot_y]
                if len(xy)>0:
                    ty = t[y==plot_y]
                    py = p[y==plot_y]
                    ts = ty[xy==plot_x]
                    if len(ts)>0:
                        timestamp = [ts[0]-100000]
                        timestamp.extend(ts)
                        polarity = py[xy==plot_x]
                        phi_max = np.max(phi_img)
                        if len(polarity)>2:
                            phi_events = np.array([phi_max*(t-np.min(timestamp))/(np.max(timestamp)-np.min(timestamp)) for t in timestamp])
                            norm_event_intensities = render_event_rate(phi_events,  polarity, c, 0, 0)
                            event_intensities[plot_y, plot_x, :] = get_event_intensity(norm_event_intensities, phi_events)
                            count+=1
                            mask[plot_y, plot_x] = 1
    num_event_angles=12
    aop, dop = get_aop_dop_from_images(event_intensities, num_event_angles)
    fillrate = count/(resolution[0]*resolution[1])
    # event_intensities = (event_intensities-np.min(event_intensities[event_intensities>0]))/(np.max(event_intensities[event_intensities>0])-np.min(event_intensities[event_intensities>0]))
    # event_intensities[event_intensities<0]=0
    return aop, dop, fillrate, mask, event_intensities

def render_event_rate(timestamp, polarity, contrast_threshold, zero_image_intensity, max_img_intensity):
    intensity = np.zeros((len(timestamp), 1))
    pol_stream = np.zeros((len(timestamp), 1))
    polarity = polarity.astype(np.int8)
    polarity[polarity==0]=-1
    intensity[0] = 0#zero_image_intensity
    eps=1e-5
    for i in range(1,len(timestamp)):
        intensity[i] = intensity[i-1] + np.sign(polarity[i-1])/(timestamp[i]-timestamp[i-1]+eps)
    intensity = np.exp(contrast_threshold*intensity)
    norm_intensities=intensity

    #---------------------decap------------
    # intensity[i] = intensity[i-1] + contrast_threshold*polarity[i-1]
    # intensity[i] = intensity[i-1] + np.sign(polarity[i-1])*np.exp(contrast_threshold*polarity[i-1])
    # norm_intensities = [(i/np.max(intensity)) for i in intensity]
    # norm_intensities = [zero_image_intensity+(i*(max_img_intensity-zero_image_intensity)) for i in norm_intensities]
    #Eq 11 from https://www.spiedigitallibrary.org/journals/optical-engineering/volume-61/issue-5/053101/Event-based-imaging-polarimeter/10.1117/1.OE.61.5.053101.pdf?SSO=1
    # intensity = np.exp(contrast_threshold*intensity)
    # norm_intensities=intensity
    # norm_intensities = [(i-np.min(intensity))/(np.max(intensity)-np.min(intensity)+eps) for i in intensity]
    # norm_intensities = [i*(max_img_intensity-zero_image_intensity) for i in norm_intensities]
    # norm_intensities = norm_intensities+ zero_image_intensity
    # norm_image_intensities = [ (i-np.min(image_intensities))/(np.max(image_intensities)-np.min(image_intensities)) for i in image_intensities]
    return np.asarray(norm_intensities)

def get_event_intensity(event_intensities, phase_angle):
    thetas = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    i_s = np.zeros((12,))
    i=0
    for t in thetas:
        # print(event_intensities[np.argmin(abs(phase_angle- thetas))])
        i_s[i] = event_intensities[np.argmin(abs(phase_angle- t))]
        i+=1
    return i_s

def estimate_normal_from_event_intensities(event_intensities, polarities):
    # each row in M = [1 cos2theta sin2theta]
    M_matrix = np.zeros((len(polarities), 3))
    M_matrix[:, 0] = 1

    I = np.zeros((len(polarities), 1))
    sample = 0
    # rand_ids = np.random.randint(0, len(polarities), size=6)
    # for idx in range(1, len(polarities), 2):
    # for idx in rand_ids:
    for idx in range(0, len(polarities)):
        # if sample>5:
        #     break
        M_matrix[sample, 1] = np.cos(np.deg2rad(2*polarities[idx]))
        M_matrix[sample, 2] = np.sin(np.deg2rad(2*polarities[idx]))
        I[sample] = event_intensities[idx]
        sample+=1

    unk = np.matmul(np.linalg.pinv(M_matrix), I)
    if unk[0]:
        dop = max(min(np.sqrt(unk[1]**2+unk[2]**2)/unk[0], 1),0)
        # dop = (max(I)-min(I))/(max(I)+min(I))
        if abs(dop)>1:
            print("LOOK AT ME, ITS TIMEE TEE")
        aop = np.arctan(unk[2],unk[1])/2
    else:
        return 0, 0
    # s_norm = np.linalg.norm(s)
    # s_norm = s[0]
    # if s_norm!=0:
    #     # s = s/s_norm
    #     s[0] = abs(s[0])
    #     dop = np.sqrt( s[1]**2  + s[2]**2)/(s[0])
    #     aop = 0.5*np.arctan2(s[2], s[1])
    #     if dop>1:
    #         return aop, 0
    return aop, dop

def processing_single_folder(object_dir, is_mituba=False, is_real=False, baseline="4"):
    if is_mituba:
        y,x,t,p, gt_normal = read_events(object_dir + '/events.h5', has_gt=True)
        list_of_image_files = sorted(glob.glob(object_dir + "/images/*.jpg"))
    elif is_real:
        x,y,t,p, _ = read_events(object_dir + '/events.h5', has_gt=False)
        sz = [720, 1280]
        gt_normal = np.load(object_dir+'/normal_gt2.npy')
        
    if baseline=='events':
        images = np.zeros((gt_normal.shape[0], gt_normal.shape[1], 12), np.float32)
        aop, dop, fr, mask, _ = get_aop_dop_from_events(x,y,p,t, [gt_normal.shape[0],gt_normal.shape[1]], images, object_dir)
        normals = ShapeFromPolarizationInfor(0, aop, dop, reflection_types=['specular'])
        normals_specular1 = normals[0]
        normals_specular2 = normals[1]



        normals_specular2[np.sum(gt_normal,axis=2)==0]=0
        normals_specular1[np.sum(gt_normal,axis=2)==0]=0
        normals_specular2[mask==0]=0
        normals_specular1[mask==0]=0

        specular2_metrics = evaluate_normals(normals_specular2, gt_normal)
        specular1_metrics = evaluate_normals(normals_specular1, gt_normal)

        return [specular1_metrics, specular2_metrics, normals_specular2, normals_specular1, gt_normal, fr]

    if baseline =="smith":
        fr = 1
        normals = np.load(object_dir+'/baseline_smith.npy')
        normals[np.sum(gt_normal,axis=2)==0]=0
        diffuse_metrics = evaluate_normals(normals, gt_normal)
        # specular_metrics = evaluate_normals(normals, gt_normal)
        return [diffuse_metrics, diffuse_metrics, normals, normals, gt_normal, fr]
    
    elif baseline == "mahmoud":
        fr = 1
        normals = np.load(object_dir+'/baseline_mahmoud.npy')
        normals[np.sum(gt_normal,axis=2)==0]=0
        diffuse_metrics = evaluate_normals(normals, gt_normal)
        specular_metrics = evaluate_normals(normals, gt_normal)
        return [diffuse_metrics, specular_metrics, normals, normals, gt_normal, fr]
    
   


def main():
    parser = argparse.ArgumentParser(
        description='Estimate surface normals from events\n',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-path_dir',  type=str,default="/tmp/snaga/D_SfP/realworld_dataset_clean/test/",help='Folder containing events and images ')
    parser.add_argument('-exp_name',  type=str,default="real",help='dataset')
    parser.add_argument('-result_dir',  type=str,default="results/",help='dir to save results')
    parser.add_argument('-baseline',  type=str, default="events", help='Is mitsuba dataset of deepsfp')

    args = parser.parse_args()
    is_mitsuba = False
    is_real = False
    if "mitsuba" in args.path_dir:
        is_mitsuba = True
        print("Evaluating SfP synthetic dataset")
    elif "real" in args.path_dir:
        is_real = True
        print("Evaluating SfP real dataset")
    else:
        print("Check the arguments")
        exit()

    objects_dir = [f.path for f in os.scandir(args.path_dir) if f.is_dir()]
    result_txt = os.path.join(args.result_dir, args.exp_name+'.txt')
    exp_result_dir = os.path.join(args.result_dir, args.exp_name+'/')
    if not os.path.exists(exp_result_dir):
        os.makedirs(exp_result_dir)
    exp_gt_dir = os.path.join(args.result_dir, 'gt/')
    if not os.path.exists(exp_gt_dir):
        os.makedirs(exp_gt_dir)
    f = open(result_txt, "w")
    metrics = ["mean_ae", "median_ae", "rmse_ae", "ae_11", "ae_22", "ae_30", "fillrate"]
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric] = []

    if is_mitsuba:
        error_sum = np.zeros((512, 512))
        mask_sum = np.zeros((512, 512))
    elif is_real:
        error_sum = np.zeros((720, 1280))
        mask_sum = np.zeros((720, 1280))
    
    for f_idx, object_dir in enumerate(objects_dir):
        msg = processing_single_folder(object_dir, is_mitsuba, is_real, args.baseline)
        specular1_metrics = msg[0]
        # specular2_metrics = msg[1]
        # normals_diffuse = msg[2]
        # normals_specular = msg[3]
        gt_normal = msg[4]
        fillrate = msg[5]
        (mean_ae, diff_color, median_ae, rmse_ae, ae_11, ae_22, ae_30, mae_map, valid_mask) = specular1_metrics
        error_sum += mae_map 
        mask_sum += valid_mask
        for idx, metric in enumerate([mean_ae, median_ae, rmse_ae, ae_11, ae_22, ae_30, fillrate]):
            f.writelines("{} {} {}\n".format(objects_dir[f_idx], metrics[idx], metric))
            metrics_dict[metrics[idx]].append(metric)
        f_idx+=1
    print("\n\n")
    for metric in metrics:
        print("{} {}: {}\n".format("average", metric, np.mean(metrics_dict[metric])))
        f.writelines("{} {}: {:.3f}\n".format("average", metric, np.mean(metrics_dict[metric])))
    f.close()

if __name__ == '__main__':
    main()
