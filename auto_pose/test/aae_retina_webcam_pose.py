import cv2
import numpy as np
import os
import argparse
import configparser

from auto_pose.ae.utils import get_dataset_path
from aae_retina_pose_estimator import AePoseEstimator



parser = argparse.ArgumentParser()
parser.add_argument("-test_config", type=str, required=False, default='test_config_webcam.cfg')
parser.add_argument("-vis", action='store_true', default=False)
args = parser.parse_args()


workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print 'Please define a workspace path:\n'
    print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
    exit(-1)


test_configpath = os.path.join(workspace_path,'cfg_eval',args.test_config)
test_args = configparser.ConfigParser()
test_args.read(test_configpath)
icp = test_args.get('ICP', 'icp')
ae_pose_est = AePoseEstimator(test_configpath)


if args.vis:
    from auto_pose.meshrenderer import meshrenderer

    ply_model_paths = [str(train_args.get('Paths','MODEL_PATH')) for train_args in ae_pose_est.all_train_args]
    cad_reconst = [str(train_args.get('Dataset','MODEL')) for train_args in ae_pose_est.all_train_args]
    
    renderer = meshrenderer.Renderer(ply_model_paths, 
                    samples=1, 
                    vertex_tmp_store_folder=get_dataset_path(workspace_path),
                    vertex_scale=float(1)) # float(1) for some models

color_dict = [(0,255,0),(0,0,255),(255,0,0),(255,255,0)] * 10




import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
labels_to_names = { 0: 'bottle',
                    1: 'box',
                    2: 'brush',
                    3: 'cabbage',
                    4: 'dolphin',
                    5: 'eggplant',
                    6: 'hedgehog',
                    7: 'lion',
                    8: 'polarbear',
                    9: 'squirrel'}

# def get_session():
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     return tf.Session(config=config)


# keras.backend.tensorflow_backend.set_session(get_session())
# model_path = '/home/robot/lxc/scripts/graspProject/detection_models/snapshots/resnet50_pascal_01_converted_to_inference.h5'
# model = models.load_model(model_path, backbone_name='resnet50')
# try:
#     model = models.convert_model(model)
# except:
#     print("model had been converted to inference before.")


import pyrealsense2 as rs
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
if icp: config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale_sr = depth_sensor.get_depth_scale() # multiplies this to get meter unit in depth map 
print("Depth Scale for RealSense SR300 is: " , depth_scale_sr)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        image = color_image
        MM_SCALE = depth_scale_sr*1000 # convert to mm
        depth_image = MM_SCALE*depth_image
        
        start = time.time()
        boxes, scores, labels = ae_pose_est.process_detection(image)
        all_pose_estimates, all_class_idcs = ae_pose_est.process_pose(boxes, labels, image, depth_img=depth_image)
        print("processing time: ", time.time() - start)

        

        if args.vis:
            bgr, depth,_ = renderer.render_many(obj_ids = [clas_idx for clas_idx in all_class_idcs],
                        W = ae_pose_est._width,
                        H = ae_pose_est._height,
                        K = ae_pose_est._camK, 
                        # R = transform.random_rotation_matrix()[:3,:3],
                        Rs = [pose_est[:3,:3] for pose_est in all_pose_estimates],
                        ts = [pose_est[:3,3] for pose_est in all_pose_estimates],
                        near = 10,
                        far = 10000,
                        random_light=False,
                        phong={'ambient':0.4,'diffuse':0.8, 'specular':0.3})

            bgr = cv2.resize(bgr,(ae_pose_est._width,ae_pose_est._height))
            
            g_y = np.zeros_like(bgr)
            g_y[:,:,1]= bgr[:,:,1]    
            im_bg = cv2.bitwise_and(image,image,mask=(g_y[:,:,1]==0).astype(np.uint8))                 
            image_show = cv2.addWeighted(im_bg,1,g_y,1,0)

            #cv2.imshow('pred view rendered', pred_view)
            for label,box,score in zip(labels,boxes,scores):
                box = box.astype(np.int32)
                xmin,ymin,xmax,ymax = box[0],box[1],box[0]+box[2],box[1]+box[3]
                print label
                cv2.putText(image_show, '%s : %1.3f' % (label,score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, color_dict[int(label)], 2)
                cv2.rectangle(image_show,(xmin,ymin),(xmax,ymax),(255,0,0),2)

            # cv2.imshow('', bgr)

            if icp:
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                # Stack both images horizontally
                image_show = np.hstack((image_show, depth_colormap))
            cv2.imshow('real', image_show)

            cv2.waitKey(1)

finally:
    # Stop streaming
    print("stoping streaming...")
    pipeline.stop()
    print("streamming stopped.")