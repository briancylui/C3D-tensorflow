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
import PIL.Image as Image
import numpy as np
import shutil
from tqdm import tqdm

def verify_images_resized(filename, resize_height=128, resize_width=171, start_index=0):
    lines = open(filename,'r').readlines()
    for video_index in tqdm(range(start_index, len(lines))):
        line = lines[video_index].strip('\n').split()
        dirname = line[0]
        resized_dirname = dirname + '_resized'
        
        if not os.path.exists(resized_dirname):
            tqdm.write('Error: directory {} does not exist'.format(resized_dirname))
            continue
        
        num_resized_frames = len([name for name in os.listdir(resized_dirname)])
        num_original_frames = len([name for name in os.listdir(dirname)])
        if num_resized_frames != num_original_frames:
            tqdm.write('Number of frames: {} = {} | resized = {}'.format(dirname, \
                num_original_frames, num_resized_frames))
            continue
        '''
        for frame in os.listdir(resized_dirname):
            image_name = os.path.join(resized_dirname, frame)
            image = Image.open(image_name)
            width, height = image.size
            if (width, height) != (resize_width, resize_height):
                tqdm.write('{}: width = {} -> {} | height = {} -> {}'.format(dirname, \
                    resize_width, width, resize_height, height))
        '''

if __name__ == '__main__':
    verify_images_resized('./list/test.list')
    print('Done verifying sizes of video frames for testing')
    verify_images_resized('./list/train.list')
    print('Done verifying sizes of video frames for training')