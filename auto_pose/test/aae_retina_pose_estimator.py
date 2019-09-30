import cv2
import numpy as np
import glob
import os
import configparser

from auto_pose.ae import factory, utils

import keras
from keras_retinanet.models import load_model, backbone
from keras_retinanet.models.retinanet import __build_anchors as build_anchors
from keras_retinanet.models.retinanet import AnchorParameters
from keras_retinanet import layers
from keras_retinanet.utils.image import preprocess_image, resize_image
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


class AePoseEstimator:
    """ """

    # Takes a configPath only!
    def __init__(self, test_configpath):

        test_args = configparser.ConfigParser()
        test_args.read(test_configpath)
        self.test_args = test_args # lxc
        workspace_path = os.environ.get('AE_WORKSPACE_PATH')

        if workspace_path == None:
            print 'Please define a workspace path:\n'
            print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
            exit(-1)

        self._camPose = test_args.getboolean('CAMERA','camPose')
        self._camK = np.array(eval(test_args.get('CAMERA','K_test'))).reshape(3,3)
        self._width = test_args.getint('CAMERA','width')
        self._height = test_args.getint('CAMERA','height')
        
    

        self._upright = test_args.getboolean('AAE','upright')
        self.all_experiments = eval(test_args.get('AAE','experiments'))

        self.class_names = eval(test_args.get('DETECTOR','class_names'))
        self.det_threshold = eval(test_args.get('DETECTOR','det_threshold'))
        self.icp = test_args.getboolean('ICP','icp')

        if self.icp:
            self._depth_scale = test_args.getfloat('DATA','depth_scale')
            print("depth scale:", self._depth_scale)
        self.all_codebooks = []
        self.all_train_args = []
        self.pad_factors = []
        self.patch_sizes = []

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        config.gpu_options.per_process_gpu_memory_fraction = test_args.getfloat('MODEL','gpu_memory_fraction')

        self.sess = tf.Session(config=config)
        set_session(self.sess)
        self.detector = load_model(str(test_args.get('DETECTOR','detector_model_path')), 
                            backbone_name=test_args.get('DETECTOR','backbone'))
        #detector = self._load_model_with_nms(test_args)



        for i,experiment in enumerate(self.all_experiments):
            full_name = experiment.split('/')
            experiment_name = full_name.pop()
            experiment_group = full_name.pop() if len(full_name) > 0 else ''
            log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
            ckpt_dir = utils.get_checkpoint_dir(log_dir)
            train_cfg_file_path = utils.get_train_config_exp_file_path(log_dir, experiment_name)
            print train_cfg_file_path
            # train_cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
            train_args = configparser.ConfigParser()
            train_args.read(train_cfg_file_path)
            self.all_train_args.append(train_args)
            self.pad_factors.append(train_args.getfloat('Dataset','PAD_FACTOR'))
            self.patch_sizes.append((train_args.getint('Dataset','W'), train_args.getint('Dataset','H')))

            self.all_codebooks.append(factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=False))
            saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=experiment_name))
            factory.restore_checkpoint(self.sess, saver, ckpt_dir)


            # if self.icp:
            #     assert len(self.all_experiments) == 1, 'icp currently only works for one object'
            #     # currently works only for one object
            #     from auto_pose.icp import icp
            #     self.icp_handle = icp.ICP(train_args)
        if test_args.getboolean('ICP','icp'):
            from auto_pose.icp import icp
            self.icp_handle = icp.ICP(test_args, self.all_train_args)


    def extract_square_patch(self, scene_img, bb_xywh, pad_factor,resize=(128,128),interpolation=cv2.INTER_NEAREST,black_borders=False):

        x, y, w, h = np.array(bb_xywh).astype(np.int32)
        size = int(np.maximum(h, w) * pad_factor)

        
        left = np.maximum(x+w//2-size//2, 0)
        right = x+w//2+size//2
        top = np.maximum(y+h//2-size//2, 0)
        bottom = y+h//2+size//2

        # left = x
        # right = x+w # without padding
        # top = y
        # bottom = y+h
        #print("left {} right {} top {} bottom {}".format(left, right, top, bottom))
        scene_crop = scene_img[top:bottom, left:right].copy()

        if black_borders:
            scene_crop[:(y-top),:] = 0
            scene_crop[(y+h-top):,:] = 0
            scene_crop[:,:(x-left)] = 0
            scene_crop[:,(x+w-left):] = 0

        scene_crop = cv2.resize(scene_crop, resize, interpolation = interpolation)

        return scene_crop

    def process_detection(self, color_img):

        H, W = color_img.shape[:2]

        pre_image = preprocess_image(color_img)
        res_image, scale = resize_image(pre_image)

        batch_image = np.expand_dims(res_image, axis=0)
        print batch_image.shape
        print batch_image.dtype
        boxes, scores, labels = self.detector.predict_on_batch(batch_image)


        valid_dets = np.where(scores[0] >= self.det_threshold)

        boxes /= scale

        scores = scores[0][valid_dets]
        boxes = boxes[0][valid_dets]
        labels = labels[0][valid_dets]

        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for box,score,label in zip(boxes, scores, labels):

            box[0] = np.minimum(np.maximum(box[0],0),W)
            box[1] = np.minimum(np.maximum(box[1],0),H)
            box[2] = np.minimum(np.maximum(box[2],0),W)
            box[3] = np.minimum(np.maximum(box[3],0),H)

            bb_xywh = np.array([box[0],box[1],box[2]-box[0],box[3]-box[1]])
            if bb_xywh[2] < 0 or bb_xywh[3] < 0:
                continue

            if label in filtered_labels: # only single instance for each class
                continue
            filtered_boxes.append(bb_xywh)
            filtered_scores.append(score)
            filtered_labels.append(label)
        return (filtered_boxes, filtered_scores, filtered_labels)



    def process_pose(self, filtered_boxes, filtered_labels, color_img, depth_img=None, camPose=None):
        normalize_pointcloud = True
        inexact_model = True
        subtract_ktrain = False
        H, W = color_img.shape[:2]

        all_pose_estimates = []
        all_class_idcs = []
        K_test = np.array(eval(self.test_args.get('CAMERA', 'K_test'))).reshape(3,3) # intrinsic of RealSense.
        Kinv = np.linalg.inv(K_test)

        for j,(box_xywh,label) in enumerate(zip(filtered_boxes,filtered_labels)):
            H_est = np.eye(4)
            try:
                clas_idx = self.class_names.index(label)
            except:
                print('%s not contained in config class_names %s', (label, self.class_names))
                continue


            det_img = self.extract_square_patch(color_img, 
                                                box_xywh, 
                                                self.pad_factors[clas_idx],
                                                resize=self.patch_sizes[clas_idx], 
                                                interpolation=cv2.INTER_LINEAR,
                                                black_borders=True)
            
            
            # len(Rs_est) = top_n
            # test_depth = f_test / f_syn * z_syn * diag_bb_ratio OR
            # test_depth = z_syn * (l_real / l_syn) where l is the diagonal length in mm.
            voc_classes =   {
                            'bottle': 0,
                            'box': 1,
                            'brush': 2,
                            'cabbage': 3,
                            'dolphin': 4,
                            'eggplant': 5,
                            'hedgehog': 6,
                            'lion': 7,
                            'polarbear': 8,
                            'squirrel': 9
                            }
            L_objects = {0:250.4, 2:327.5, 3:260., 4:318.7, 5:277.4, 6:266.7, 
                         8:308.7, 7:345.0, 9:294.6}
            L_object = L_objects[label]
            print("label:", label)
            print("L_object:", L_object)
            

            Rs_est, ts_est = self.all_codebooks[clas_idx].auto_pose6d(self.sess, 
                                                                        det_img, 
                                                                        box_xywh, 
                                                                        self._camK,
                                                                        1, 
                                                                        self.all_train_args[clas_idx], 
                                                                        upright=self._upright,
                                                                        subtract_ktrain=subtract_ktrain,
                                                                        inexact_model = inexact_model,
                                                                        L_object = L_object
                                                                        )
            print("estimated translation:", ts_est[0])

            if depth_img is not None:
                # look at real x,y,z, currently support one object only.
                x, y, w, h = np.array(box_xywh).astype(np.int32)
                R_est, t_est = Rs_est[0], ts_est[0]
                u_center, v_center = x+w//2, y+h//2
                z_real = depth_img[v_center, u_center]
                translation_real = z_real*Kinv.dot([u_center, v_center, 1])
                print("real translation: ", translation_real[:3])

            # calculate the diagonal length of synthetic object in real world.
            # assume diagonal points A and B have the same depth z.

            R_est = Rs_est.squeeze()
            t_est = ts_est.squeeze()

            if self.icp:
                assert H == depth_img.shape[0]
                
                print("mean real depth:", np.mean(depth_img))
                interpo_method = cv2.INTER_NEAREST
                # interpo_method = cv2.INTER_LINEAR
                depth_crop = self.extract_square_patch(depth_img, 
                                                    box_xywh,
                                                    self.pad_factors[clas_idx],
                                                    resize=self.patch_sizes[clas_idx], 
                                                    interpolation=interpo_method) / self._depth_scale
                print("mean real crop depth:", np.mean(depth_crop))

                x, y, w, h = np.array(box_xywh).astype(np.int32)
                print(x,y,w,h)
                print("mean real crop depth:", np.mean(depth_img[y:y+h, x:x+w]))

                R_est_auto = R_est.copy()
                t_est_auto = t_est.copy()
                # first refine: Rotation and Translation.
                # depth only set to False by lxc.
                R_est, t_est = self.icp_handle.icp_refinement(depth_crop, R_est, t_est, self._camK, (W,H), clas_idx=clas_idx, depth_only=False, normalize_pointcloud=normalize_pointcloud)
                print("icp refine 1 t_est:", t_est)
                # second refine: depth only.
                # _, ts_est = self.all_codebooks[clas_idx].auto_pose6d(self.sess, 
                #                                                             det_img, 
                #                                                             box_xywh, 
                #                                                             self._camK,
                #                                                             1, 
                #                                                             self.all_train_args[clas_idx], 
                #                                                             upright=self._upright,
                #                                                             depth_pred=t_est[2],
                #                                                             subtract_ktrain=subtract_ktrain,
                #                                                             inexact_model=False,
                #                                                             L_object = L_object 
                #                                                             )
                t_est = ts_est.squeeze()
                print("codebook refine t_est:", t_est)
                # third refine: Rotation only.
                # R_est, _ = self.icp_handle.icp_refinement(depth_crop, R_est, ts_est.squeeze(), self._camK, (W,H), clas_idx=clas_idx, no_depth=True, normalize_pointcloud=normalize_pointcloud)
                # depth_crop = self.extract_square_patch(depth_img, 
                #                                     box_xywh,
                #                                     self.pad_factors[clas_idx],
                #                                     resize=self.patch_sizes[clas_idx], 
                #                                     interpolation=cv2.INTER_NEAREST)
                # R_est, t_est = self.icp_handle.icp_refinement(depth_crop, R_est, t_est, self._camK, (W,H))

                H_est[:3,3] = t_est / self._depth_scale #mm / m
            else:
                H_est[:3,3] = t_est

            H_est[:3,:3] = R_est
            print 'translation from camera: ',  H_est[:3,3]

            if self._camPose:
                H_est = np.dot(camPose, H_est)           

            all_pose_estimates.append(H_est)
            all_class_idcs.append(clas_idx)


        return (all_pose_estimates, all_class_idcs)


    def _load_model_with_nms(self, test_args):
        """ This is mostly copied fomr retinanet.py """

        backbone_name = test_args.get('DETECTOR','backbone')
        print backbone_name
        print test_args.get('DETECTOR','detector_model_path')
        model = keras.models.load_model(
                str(test_args.get('DETECTOR','detector_model_path')),
                custom_objects=backbone(backbone_name).custom_objects
                )

        # compute the anchors
        features = [model.get_layer(name).output
                for name in ['P3', 'P4', 'P5', 'P6', 'P7']]
        anchors  = build_anchors(AnchorParameters.default, features)

        # we expect the anchors, regression and classification values as first
        # output
        print len(model.outputs)
        regression     = model.outputs[0]
        classification = model.outputs[1]
        print classification.shape[1]
        print regression.shape

        # "other" can be any additional output from custom submodels,
        # by default this will be []
        other = model.outputs[2:]

        # apply predicted regression to anchors
        boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
        boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

        # filter detections (apply NMS / score threshold / select top-k)
        #detections = layers.FilterDetections(
        #        nms=True,
        #        name='filtered_detections',
        #        nms_threshold = test_args.getfloat('DETECTOR','nms_threshold'),
        #        score_threshold = test_args.getfloat('DETECTOR','det_threshold'),
        #        max_detections = test_args.getint('DETECTOR', 'max_detections')
        #        )([boxes, classification] + other)        
        detections = layers.filter_detections.filter_detections(
                boxes=boxes,
                classification=classification,
                other=other,
                nms=True,
                nms_threshold = test_args.getfloat('DETECTOR','nms_threshold'),
                score_threshold = test_args.getfloat('DETECTOR','det_threshold'),
                max_detections = test_args.getint('DETECTOR', 'max_detections')
                )

        outputs = detections

        # construct the model
        return keras.models.Model(
                inputs=model.inputs, outputs=outputs, name='retinanet-bbox')