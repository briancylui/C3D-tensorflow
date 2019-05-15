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

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import PIL.Image as Image
import numpy as np
import shutil
from tqdm import tqdm
import time
import subprocess


def resize_images(filename, resize_height=128, resize_width=171, start_index=404):
    lines = open(filename,'r').readlines()
    for video_index in tqdm(range(start_index, len(lines))):
        line = lines[video_index].strip('\n').split()
        dirname = line[0]
        resized_dirname = dirname + '_resized'
        
        if os.path.exists(resized_dirname):
            num_resized_frames = len([name for name in os.listdir(resized_dirname)])
            num_original_frames = len([name for name in os.listdir(dirname)])
            if num_resized_frames < num_original_frames:
                shutil.rmtree(resized_dirname)
                os.mkdir(resized_dirname)
            else:
                continue
        else:
            os.mkdir(resized_dirname)
        for parent, dirnames, filenames in os.walk(dirname):
            for filename in filenames:
                image_name = os.path.join(parent, filename)
                image = Image.open(image_name)
                resized_image = image.resize((resize_width, resize_height), Image.LANCZOS)
                resized_image.save(os.path.join(resized_dirname, filename))

if __name__ == '__main__':
    # resize_images('./list/test.list')
    # print('Done resizing video frames for testing')
    resize_images('./list/train.list')
    print('Done resizing video frames for testing')