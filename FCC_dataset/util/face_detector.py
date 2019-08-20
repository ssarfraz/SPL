

#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image.  In
#   particular, it shows how you can take a list of images from the command
#   line and display each on the screen with red boxes overlaid on each human
#   face.
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
#   following command:
#       ./face_detector.py ../examples/faces/*.jpg
#
#   This face detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme.  This type of object detector
#   is fairly general and capable of detecting many types of semi-rigid objects
#   in addition to human faces.  Therefore, if you are interested in making
#   your own object detectors then read the train_object_detector.py example
#   program.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import sys,os, glob

import dlib
import cv2

detector = dlib.get_frontal_face_detector()
print(sys.argv)

#win = dlib.image_window()

files = sys.argv[1:]
print(sys.argv[1])
print(len(files))
print(len(sys.argv))
for f in files:
    print("Processing file: {}".format(f))
    #loads image into numpy array
    img = dlib.load_rgb_image(f)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)

    print("Number of faces detected: {}".format(len(dets)))
    if len(dets)>0:
        for i, d in enumerate(dets[0:1]):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom()))
            img = img[max(0, d.top()): min(d.bottom(), img.shape[0]),
                            max(0, d.left()): min(d.right(), img.shape[1])]

        if img.shape[0] >400 and img.shape[1] >400:
            img2 = cv2.resize(img,(512,512))
            #win.clear_overlay()
            #win.set_image(img)
            k = f.split("/")
            path = "/cvhci/users/cseibold/paper/transfer/nomakeup"
            f_path = "/cvhci/users/cseibold/paper/transfer/nomakeup"+k[-1][:-4]+"_512_cropped.jpg"
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(f_path, cv2.cvtColor(img2,cv2.COLOR_RGB2BGR))
        img = cv2.resize(img,(256,256))
        #win.clear_overlay()
        #win.set_image(img)
        k = f.split("/")
        path = "/cvhci/users/cseibold/paper/transfer/nomakeup"
        f_path = "/cvhci/users/cseibold/paper/transfer/nomakeup"+k[-1][:-4]+"_256_cropped.jpg"
        all_path = "/cvhci/users/cseibold/paper/transfer/nomakeup"
        f_all_path = "/cvhci/users/cseibold/paper/transfer/nomakeup"+k[-1][:-4]+"_256_cropped.jpg"
        if not os.path.exists(path):
            os.makedirs(path)
	
        if not os.path.exists(all_path):
            os.makedirs(all_path)
        cv2.imwrite(f_all_path, cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        cv2.imwrite(f_path, cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    if os.path.exists(f):
        os.remove(f)
    #win.add_overlay(dets)
    #dlib.hit_enter_to_continue()


# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.
"""if (len(sys.argv[1:]) > 0):
    img = dlib.load_rgb_image(sys.argv[1])
    dets, scores, idx = detector.run(img, 1, -1)
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))
"""
