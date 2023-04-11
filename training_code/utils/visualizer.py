import numpy as np
import os
import numpy as np
from PIL import Image
import ntpath
# from scipy.misc import imresize
import cv2

def save_image(image_numpy, image_path, rs=False):
    cv2.imwrite(image_path, image_numpy)

def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor  #output image range (-1,1)
    return image_numpy.astype(imtype)

def tensor2norm(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1.) * 0.5*255.0  #output image range (-1,1)
    return image_numpy.astype(imtype)

# save image to the disk
def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = str(image_path)#os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


def markdown_visualizer(save_image_dir='./results/dialunet_fp32_spwinter_drop0_1/249', num=10, mae=[]):
    f= open("{}/a_visualizer.md".format(save_image_dir), 'w')
    f.writelines("|Index| mAE |Input|Pred|GT|\n")
    f.writelines("|:---:|:---:|:---:|:--:|:--:|\n")
    idx = 0
    for i in range(num):
        f.writelines("|{:03d}|{:.2f}|![input](im{}_in.jpg)|![pred](im{}_pred.jpg)|![gt](im{}_gt.jpg)|\n".format(i, mae[idx], i, i,  i))
        idx+=1

def markdown_visualizer_img(save_image_dir='', num=10, mae=[]):
    f= open("{}/a_visualizer.md".format(save_image_dir), 'w')
    f.writelines("|Index| mAE |Input|Pred| GT |\n")
    f.writelines("|:---:|:---:|:---:|:--:|:--:|\n")
    idx = 0
    for i in range(num):
        f.writelines("|{:03d}|{:.2f}|![input](im{}_in.jpg)|![pred](im{}_pred.jpg)|![gt](im{}_gt.jpg)||\n".format(i, mae[idx], i, i, i))
        idx+=1

def markdown_visualizer_test(imgnames=[], output_folder = './'):
    f= open("{}/a_visualizer.md".format(output_folder), 'w')
    f.writelines("|Image |Image|Image|Image|\n")
    f.writelines("|:---:|:---:|:---:|:--:|\n")
    for i in range(len(imgnames)//4):
        f.writelines("|![input]({})|![input]({})|![input]({})|![input]({})|\n".format(imgnames[4*i], imgnames[4*i+1], imgnames[4*i+2],  imgnames[4*i+3]))

        f.writelines("|{}|{}|{}|{}|\n".format( "/".join(imgnames[4*i+0].split("/")[-3:]),
                                            "/".join(imgnames[4*i+1].split("/")[-3:]),
                                            "/".join(imgnames[4*i+2].split("/")[-3:]),
                                            "/".join(imgnames[4*i+3].split("/")[-3:]),
                                              ))

