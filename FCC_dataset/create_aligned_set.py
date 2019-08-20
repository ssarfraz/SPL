import numpy as np
import cv2,sys
import scipy
import random,pdb
import glob
import os
import ntpath
from util.landmarkWarp import warp_im

def main(pathA, pathB, outpath='./pairs', pairs=10, img_size=256):
    filename_A= sorted(glob.glob(pathA+'/*'))
    filename_B= sorted(glob.glob(pathB+'/*'))
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    
    for i in range(len(filename_A)):
        for j in range(pairs):

            imA=cv2.imread( filename_A[i])
            imA=cv2.resize(imA,(img_size,img_size), interpolation=cv2.INTER_CUBIC)       

            name = filename_B[np.random.randint(0,len(filename_B))]

            print('processing Image ', filename_A[i],name)

            imB=cv2.imread(name)        
            imB_ini=cv2.resize(imB,(img_size,img_size), interpolation=cv2.INTER_CUBIC)       
            imB,a,b=warp_im(imB_ini, imA)
            
            tmp=imB.copy().astype('uint8')

            mask = np.ones((img_size,img_size,3),dtype='uint8')
            mask = mask*255
            mask[imB==0]= 0 

            tmp[imB==0]=imA[imB==0]
            tmp[imB!=0]=imB[imB!=0]*255   

            new_img = np.ones((img_size,img_size*2,3),dtype='uint8')
            new_img[:,:img_size,:] = imA
            new_img[:,img_size:img_size*2,:] = tmp
            tmp = np.int_(tmp)
            tmp = tmp.astype('uint8')

            sname =  filename_A[i].split('/')[-1].split('.jpg')[0] +name.split('/')[-1].split('.jpg')[0] + '.jpg'

            cv2.imwrite(os.path.join(outpath,sname), new_img)
    
    
if __name__ == '__main__':
    pathB =  sys.argv[1]
    pathA =  sys.argv[2]
    outpath = sys.argv[3]
    pairs = sys.argv[4]
    img_size = sys.argv[5]
    main(pathA, pathB, outpath, pairs = int(pairs), img_size = int(img_size))