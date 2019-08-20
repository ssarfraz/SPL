# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:40:07 2018

@author: ssarfraz
"""
from imutils import face_utils
import numpy as np
from skimage import transform
#from skimage import io
import dlib
import sys
import cv2
#import tensorflow as tf
#from PIL import Image

def warp_it(image, src, dst):
    # estimate transformation parameters
	tform = transform.estimate_transform('piecewise-affine', src, dst) #similarity piecewise-affine

	np.allclose(tform.inverse(tform(src)), src)

    # warp image using the estimated transformation
	im= transform.warp(image, inverse_map=tform.inverse, mode='constant', cval=0) #tform.inverse
	return im


def get_face_border(image, pred_path='./util/shape_predictor_68_face_landmarks.dat'):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pred_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        face = np.zeros((70,2))
        face[0:67] = shape[0:67]
        
        face[68] = [0,0]
        face[69] = [image.shape[1],0]

        face.reshape((-1,1,2))

    return np.int_(face)


def get_landmarks(img):
    predictor_path = './util/shape_predictor_68_face_landmarks.dat' # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    dets = detector(img, 0)
#
    if len(dets) == 0:
        rect = dlib.rectangle(0,0,img.shape[0], img.shape[1])
        shape = predictor(img, rect)
        xy = _shape_to_np(shape)
    else:
       for rect in dets:
        shape = predictor(img, rect)
        xy = _shape_to_np(shape)
    return xy		

def _shape_to_np(shape):
    xy = []
    for i in range(68):
        xy.append((shape.part(i).x, shape.part(i).y))
    xy = np.asarray(xy, dtype='float32')
    return xy


def warp_im(src_im, dst_im):
     src= get_landmarks(src_im)
     dst= get_landmarks(dst_im)
     wim=warp_it(src_im, src, dst)
 		
     return wim, src, dst

if __name__ == '__main__':
    filename_1 = sys.argv[1]
    filename_2 = sys.argv[2]
    src_im = cv2.resize(cv2.imread(filename_1), (512, 512))
    
    src_im[src_im<30] = 0
    dst_im = cv2.resize(cv2.imread(filename_2), (512, 512))
				
    wim, src, dst = warp_im(src_im, dst_im)
    #mat=wim>0
    				
    tmp=wim.copy().astype('uint8')
    tmp[wim==0]=dst_im[wim==0]
    tmp[wim!=0]=wim[wim!=0]*255			

    for (x, y) in src:
        cv2.circle(src_im, (x, y), 1, (0, 0, 255), -1)
        
    for (x, y) in dst:
        cv2.circle(dst_im, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("reference", tmp)
    cv2.imshow("src", src_im)
    cv2.imshow("dst", dst_im)
    cv2.imshow("warped", wim)
    				
    cv2.waitKey(0)				
