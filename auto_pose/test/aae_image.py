import cv2
import tensorflow as tf
import numpy as np
import glob
import os
import time
import argparse
import configparser

from auto_pose.ae import factory, utils



parser = argparse.ArgumentParser()
parser.add_argument("experiment_name")
parser.add_argument("-f", "--file_str", required=True, help='folder or filename to image(s)')
# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

file_str = arguments.file_str
if os.path.isdir(file_str):
    files = sorted(glob.glob(os.path.join(str(file_str),'*.png'))+glob.glob(os.path.join(str(file_str),'*.jpg')))
else:
    files = [file_str]

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print 'Please define a workspace path:\n'
    print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
    exit(-1)
log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
ckpt_dir = utils.get_checkpoint_dir(log_dir)

codebook, dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)


with tf.Session() as sess:
  
    factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir, at_step=None) # try at_step = 2000
    # Rpred_list = [] #lxc
    for file in files:
        im = cv2.imread(file)
        im = cv2.resize(im,(128,128))

        R = codebook.nearest_rotation(sess, im, top_n=3)
        print R
        # Rpred_list.append(R.reshape(1,-1)) #lxc
        pred_view = dataset.render_rot( R,downSample = 1)
    
        cv2.imshow('resized img', cv2.resize(im/255.,(256,256)))
        cv2.imshow('pred_view', cv2.resize(pred_view,(256,256)))
        if cv2.waitKey(0):
            continue
    # np.savetxt('/media/lxc/6044A61C44A5F546/LXC/Graduate2Spring/Grasp/aae_workspace/experiments/exp_group/dolphin_stl/Rpred.txt', np.array(Rpred_list).squeeze(), fmt='%5.5f', delimiter=',')



