#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 21:55:27 2018

@author: kausic
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:24:47 2018

@author: kausic
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

root="/home/kausic/Desktop/My_research/dataset/sunrgbd/SUNRGBD"
save_location="/home/kausic/ASU_MS/SML/project/sunrgbd_images/"
data_file=open(save_location+"data.txt",'w')
count=0

for dirn,subn,fileList in tqdm(os.walk(root,True),desc="Files recorded"):
    if('annotation' in subn):
        #print (subn)
        (_,__,files)=os.walk(dirn +'/image/').__next__()
        image_path=dirn+'/image/'+files[0]
        (_,__,files)=os.walk(dirn +'/depth/').__next__()
        depth_path=dirn+'/depth/'+files[0]
        scene_file=open(dirn+'/scene.txt')
        scene=scene_file.read()
        scene_file.close()
        rgb_img=cv2.imread(image_path)
        depth_img=cv2.imread(depth_path)
        if rgb_img is None or depth_img is None:
            continue
        final_string="img_{0:05d} ".format(count)
        img_name="rgb_img_{0:05d}.jpg".format(count)
        depth_name="depth_img_{0:05d}.jpg".format(count)
        final_string+=scene
        data_file.write(final_string+'\n')
        cv2.imwrite(save_location+img_name,rgb_img)
        cv2.imwrite(save_location+depth_name,depth_img)
        count+=1
data_file.close()
