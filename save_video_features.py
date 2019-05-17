# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import os
import sys
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import h5py
from tqdm import tqdm
import c3d_model_ucfcrime

# Hyperparameters
CROP_SIZE = 112
NUM_FRAMES_PER_CLIP = 16
CLIP_INCREMENT = 8
NUM_SEGMENTS_PER_VIDEO = 32
GPU_NUM = 4
CHANNELS = 3
MODEL_NAME = './models/c3d_ucf101_finetune_whole_iter_20000_TF.model'
model = c3d_model_ucfcrime
FEATURE_FILE = './ucfcrime_c3d_features.h5'
LISTS = ['./list/test.list']
SEED = 171

clip_mean = np.expand_dims(np.load('./crop_mean.npy').reshape([NUM_FRAMES_PER_CLIP, \
    CROP_SIZE, CROP_SIZE, CHANNELS]), axis=0)

def get_random_crop(image_np, crop_size=CROP_SIZE):
    height, width, channels = image_np.shape
    # Assumes: height, width > crop_size
    # Otherwise: refer to https://github.com/tqvinhcs/C3D-tensorflow/blob/master/m_video.py
    start_index_height = np.random.randint(height - crop_size)
    start_index_width = np.random.randint(width - crop_size)
    return image_np[start_index_height:start_index_height + crop_size, \
        start_index_width:start_index_width + crop_size, :]

def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:%d' % cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var

def get_image_batch_for_segment(video_path, frames_list, num_frames_per_clip=NUM_FRAMES_PER_CLIP, \
    clip_increment=CLIP_INCREMENT):
    batch = []
    num_frames = len(frames_list)
    num_clips = (num_frames - num_frames_per_clip) // clip_increment + 1
    for clip_index in range(0, num_clips):
        start_frame_index = clip_index * clip_increment
        clip = []
        for frame_index in range(start_frame_index, start_frame_index + num_frames_per_clip):
            image = Image.open(os.path.join(video_path, frames_list[frame_index]))
            image_np = np.array(image) # image.size (W, H) -> image_data.shape (H, W, C)
            # Randomly crop image_np (H, W, C) into (crop_size, crop_size, C)
            # image_cropped = get_random_crop(image_np)
            clip.append(np.expand_dims(image_np, axis=0)) # (H, W, C) -> (1, H, W, C)
        clip = np.concatenate(clip) # (F, H, W, C)
        # Optional: Horizontally flip the entire clip w.p. 50% by uncommenting:
        # if np.random.random() < 0.5: clip = np.flip(clip, axis=2)
        batch.append(np.expand_dims(clip, axis=0)) # (F, H, W, C) -> (1, F, H, W, C)
    batch = np.concatenate(batch) # (B, F, H, W, C); B = number of clips in video (1 batch)

    return batch

def get_segment_features(video_path, frames_list, num_frames_per_clip=NUM_FRAMES_PER_CLIP, \
    clip_increment=CLIP_INCREMENT):
    clip_features = []
    num_frames = len(frames_list)
    if num_frames < num_frames_per_clip:
        return None
    
    clips = get_image_batch_for_segment(video_path, frames_list, NUM_FRAMES_PER_CLIP, \
        CLIP_INCREMENT)

    images_placeholder = tf.placeholder(tf.float32, shape=clips.shape)
    mean_placeholder = tf.placeholder(tf.float32, shape=clip_mean.shape)
    with tf.variable_scope('var_name', reuse=tf.AUTO_REUSE) as var_scope:
        weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001)
                }
        biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
                'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0)
                }
    
    features = []
    batch_size = clips.shape[0] // GPU_NUM
    for gpu_index in range(0, GPU_NUM):
        with tf.device('/gpu:%d' % gpu_index):
            batch = images_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size,:,:,:,:]
            
            cropped = tf.image.random_crop(batch, [batch_size, NUM_FRAMES_PER_CLIP, CROP_SIZE, CROP_SIZE, CHANNELS])
            cropped_zero_mean = tf.subtract(cropped, mean_placeholder)
            feature = model.inference_c3d(cropped_zero_mean, 0.6, batch_size, weights, biases)
            features.append(feature) # (B / GPU_NUM, 4096)
    features = tf.concat(features, axis=0) # (B, 4096)
    features = tf.reduce_mean(features, axis=0) # (4096,)

    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Create a saver for writing training checkpoints.
    saver.restore(sess, MODEL_NAME)

    segment_features = features.eval(
        session=sess,
        feed_dict={images_placeholder: clips, mean_placeholder: clip_mean}
        )

    return segment_features


def save_video_features(filename, num_segments_per_video=NUM_SEGMENTS_PER_VIDEO, start_index=0, end_index=340):
    lines = open(filename, 'r').readlines()
    tf.random.set_random_seed(SEED)

    for video_index in tqdm(range(start_index, end_index)):
        line = lines[video_index].strip('\n').split()
        dirname = line[0]
        resized_dirname = dirname + '_resized'
        feature_name = dirname + '.npy'

        if os.path.exists(feature_name):
            continue

        frames = os.listdir(resized_dirname)
        num_frames = len(frames)
        num_frames_per_segment = num_frames // num_segments_per_video
        video_features = []
        for segment_index in range(num_segments_per_video):
            if num_frames_per_segment >= NUM_FRAMES_PER_CLIP:
                frames_list = frames[num_frames_per_segment * \
                    segment_index:num_frames_per_segment * (segment_index + 1)]
            else:
                frames_list = frames[num_frames_per_segment * \
                    segment_index:num_frames_per_segment * segment_index + NUM_FRAMES_PER_CLIP]
             
            segment_features = get_segment_features(resized_dirname, frames_list)
            if segment_features is not None:
                video_features.append(np.expand_dims(segment_features, axis=0)) # (1, 4096)
        video_features = np.concatenate(video_features) # (NUM_SEGMENTS_PER_VIDEO, 4096)

        # Saves feature
        np.save(feature_name, video_features)

if __name__ == '__main__':
    # Suppresses INFO print-out statements
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    for listname in LISTS:
        save_video_features(listname, start_index=int(sys.argv[1]), end_index=int(sys.argv[2]))
        print('Done saving features for videos in {}'.format(listname))