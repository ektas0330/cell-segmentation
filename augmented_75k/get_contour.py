# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:10:01 2019

@author: SMARTS_Station
"""

#cell segmentation masks post processing
import cv2,fnmatch,os
if not os.path.exists('./test/contours/'):
    os.makedirs('./test/contours/')
#import numpy as np
test_src_files = fnmatch.filter(os.listdir('./test/src/'), '*.jpg')

for filename in test_src_files:
    img = cv2.imread('./test/src/' + filename)
    imgmask = cv2.imread('./test/predict/' + os.path.splitext(filename)[0]+'.png')
    channel1 = imgmask[:,:,1] #beads
    channel2 = imgmask[:,:,2]
    ret1,thresh1 = cv2.threshold(channel1,69,255,cv2.THRESH_BINARY) #experimentally set thresholds
    _ , contours1, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    showimg = cv2.drawContours(img, contours1, -1, (0,255,0),2)
    ret2,thresh2 = cv2.threshold(channel2,80,255,cv2.THRESH_BINARY) #experimentally set thresholds
    _, contours2, hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    showimg = cv2.drawContours(showimg, contours2, -1, (0,0,255),2)

    cv2.imwrite('./test/contours/'+ os.path.splitext(filename)[0]+'_contour.png', showimg)
