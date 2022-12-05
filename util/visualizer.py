import numpy as np
import os
import ntpath
import time
from . import util
from scipy.misc import imresize


# save image to the disk
def save_images(images, names, image_path, test=False, opt=None, aspect_ratio=1.0, width=256):
    name = ntpath.basename(image_path)

    if test:  # test mode
        path = 'images/'
        if not os.path.exists(path):
            os.makedirs(path)
        image_dir = path
    elif opt is not None:  # validation mode
        image_dir = os.path.join(opt.checkpoints_dir, opt.name, 'validation_logs')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

    for label, im_data in zip(names, images):
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        util.save_image(im, save_path)