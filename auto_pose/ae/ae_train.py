 # -*- coding: utf-8 -*-
import os
import configparser
import argparse
import numpy as np
import signal
import shutil
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import progressbar
import tensorflow as tf

import ae_factory as factory
import utils as u


def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print 'Please define a workspace path:\n'
        print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
        exit(-1)

    gentle_stop = np.array((1,), dtype=np.bool)
    gentle_stop[0] = False
    def on_ctrl_c(signal, frame):
        gentle_stop[0] = True
    signal.signal(signal.SIGINT, on_ctrl_c)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("-d", action='store_true', default=False)
    parser.add_argument("-gen", action='store_true', default=False)
    parser.add_argument("-augdata", action='store_true', default=False) # by lxc
    arguments = parser.parse_args()

    full_name = arguments.experiment_name.split('/')
    
    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''
    
    debug_mode = arguments.d
    generate_data = arguments.gen
    augment_data =  arguments.augdata # by lxc

    cfg_file_path = u.get_config_file_path(workspace_path, experiment_name, experiment_group)
    log_dir = u.get_log_dir(workspace_path, experiment_name, experiment_group)
    checkpoint_file = u.get_checkpoint_basefilename(log_dir)
    ckpt_dir = u.get_checkpoint_dir(log_dir)
    train_fig_dir = u.get_train_fig_dir(log_dir)
    dataset_path = u.get_dataset_path(workspace_path)
    
    print("cfg_file_path:{}".format(cfg_file_path))
    print("log_dir:{}".format(log_dir))
    print("checkpoint_file:{}".format(checkpoint_file))
    print("ckpt_dir:{}".format(ckpt_dir))
    print("train_fig_dir:{}".format(train_fig_dir))
    print("dataset_path:{}".format(dataset_path))

    if not os.path.exists(cfg_file_path):
        print 'Could not find config file:\n'
        print '{}\n'.format(cfg_file_path)
        exit(-1)
        
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(train_fig_dir):
        os.makedirs(train_fig_dir)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        



    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    shutil.copy2(cfg_file_path, log_dir)

    with tf.variable_scope(experiment_name):
        dataset = factory.build_dataset(dataset_path, args)
        queue = factory.build_queue(dataset, args)
        encoder = factory.build_encoder(queue.x, args, is_training=True)
        decoder = factory.build_decoder(queue.y, encoder, args, is_training=True)
        ae = factory.build_ae(encoder, decoder, args)
        codebook = factory.build_codebook(encoder, dataset, args)
        train_op = factory.build_train_op(ae, args)
        saver = tf.train.Saver(save_relative_paths=True)

    num_iter = args.getint('Training', 'NUM_ITER') if not debug_mode else np.iinfo(np.int32).max
    save_interval = args.getint('Training', 'SAVE_INTERVAL')
    model_type = args.get('Dataset', 'MODEL')

    if model_type=='dsprites': # if use deepmind dsprites-dataset
        dataset.get_sprite_training_images(args)
    else:
        dataset.get_training_images(dataset_path, args, experiment_name, augment_data = augment_data)
        dataset.load_bg_images(dataset_path, experiment_name)


    if augment_data:
        print 'finished generating AUGMENTED synthetic training data for ' + experiment_name
        print 'exiting...'
        exit()


    if generate_data:
        print 'finished generating synthetic training data for ' + experiment_name
        print 'exiting...'
        exit()

    widgets = ['Training: ', progressbar.Percentage(),
         ' ', progressbar.Bar(),
         ' ', progressbar.Counter(), ' / %s' % num_iter,
         ' ', progressbar.ETA(), ' ']
    bar = progressbar.ProgressBar(maxval=num_iter,widgets=widgets)


    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.9)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:

        chkpt = tf.train.get_checkpoint_state(ckpt_dir)
        if chkpt and chkpt.model_checkpoint_path:
            saver.restore(sess, chkpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        merged_loss_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(ckpt_dir, sess.graph)
        
                
        if not debug_mode:
            print 'Training with %s model' % args.get('Dataset','MODEL'), os.path.basename(args.get('Paths','MODEL_PATH'))
            bar.start()
        
        queue.start(sess)
        save_image_lxc = False
        for i in xrange(ae.global_step.eval(), num_iter):
            if not debug_mode:
                sess.run(train_op)
                # if save_image_lxc: # why save image with 0 byte?
                #     this_x, this_y = sess.run([queue.x, queue.y])
                #     cv2.imwrite('/home/lxc/Desktop/aae_ws/experiments/exp_group/tejani_05/this_x.jpg', this_x)
                #     cv2.imwrite('/home/lxc/Desktop/aae_ws/experiments/exp_group/tejani_05/this_y.jpg', this_y)
                # save_image_lxc = False
                if i % 10 == 0:
                    loss = sess.run(merged_loss_summary)
                    summary_writer.add_summary(loss, i)

                bar.update(i)
                if (i+1) % save_interval == 0:
                    saver.save(sess, checkpoint_file, global_step=ae.global_step)

                    this_x, this_y = sess.run([queue.x, queue.y])
                    reconstr_train = sess.run(decoder.x,feed_dict={queue.x:this_x})
                    train_imgs = np.hstack(( u.tiles(this_x, 4, 4), u.tiles(reconstr_train, 4,4),u.tiles(this_y, 4, 4)))
                    cv2.imwrite(os.path.join(train_fig_dir,'training_images_%s.png' % i), train_imgs*255)
            else:
                
                this_x, this_y = sess.run([queue.x, queue.y])
                reconstr_train = sess.run(decoder.x,feed_dict={queue.x:this_x})
                '''
                this_x: augmented image of object, 
                reconstr_train: reconstruction image of object,
                this_y: original image of object               
                '''
                cv2.imshow('sample batch', np.hstack(( u.tiles(this_x, 3, 3), u.tiles(reconstr_train, 3,3),u.tiles(this_y, 3, 3))) )
                k = cv2.waitKey(0)
                if k == 27:
                    break

            if gentle_stop[0]:
                break

        queue.stop(sess)
        if not debug_mode:
            bar.finish()
        if not gentle_stop[0] and not debug_mode: # after finishing training
            print 'To create the embedding run:\n'
            print 'ae_embed {}\n'.format(full_name)

if __name__ == '__main__':
    main()
    
