from __future__ import division
from __future__ import print_function
import os
import sys
from cv2 import (add,
                 bilateralFilter,
                 bitwise_and,
                 boundingRect,
                 Canny,
                 CHAIN_APPROX_NONE,
                 CHAIN_APPROX_SIMPLE,
                 circle,
                 connectedComponents,
                 convertScaleAbs,
                 contourArea,
                 COLOR_BGR2GRAY,
                 cvtColor,
                 dilate,
                 DIST_L2,
                 distanceTransform,
                 drawContours,
                 erode,
                 findContours,
                 imread,
                 IMREAD_GRAYSCALE,
                 imshow,
                 merge,
                 MORPH_OPEN,
                 morphologyEx,
                 rectangle,
                 RETR_TREE,
                 subtract,
                 threshold,
                 THRESH_BINARY,
                 THRESH_BINARY_INV,
                 THRESH_OTSU,
                 watershed,
                 xfeatures2d)
from numpy import (int32,
                   ones,
                   uint8,
                   zeros)
from sklearn import (cross_validation,
                     grid_search,
                     metrics,
                     svm)
from pixel.detector import OpenCVDetector
from pixel.utility import (pixel_to_lonlat,
                           cut_keypoint,
                           write_image,
                           plot_image)

#
# Technique 1 -  Marker-based image segmentation using watershed algorithm
#

# Open image files and intiatlize SURF detector
scene = imread('/Users/tstavish/Data/surf2pixelSVM/harbor.tif',
               IMREAD_GRAYSCALE)
scene_3band = imread('/Users/tstavish/Data/harbor/harbor3.tif')
detector = xfeatures2d.SURF_create(400)
keypoints, descriptors = detector.detectAndCompute(scene, None)

# Loop through keypoints
no_of_kp = 10
keypoint_cut = []
keypoint_cut_3band = []
for kp in keypoints[:no_of_kp]:
    print('angle = ', kp.angle)
    print('class_id = ', kp.class_id)
    print('octave = ', kp.octave)
    print('x, y = ', kp.pt[0], kp.pt[1])
    print('response = ', kp.response)
    print('size = ', kp.size)
    keypoint_cut.append(cut_keypoint(int(kp.pt[0]), int(kp.pt[1]),
                        kp.size, scene, False))
    keypoint_cut_3band.append(cut_keypoint(int(kp.pt[0]), int(kp.pt[1]),
                              kp.size, scene_3band, False))
print('Number of keypoints = ', len(keypoints))

# Examine the first keypoint
img = keypoint_cut[0]
# write_image(img, 'keypoint_1band_orig.tif')
img_3b = keypoint_cut_3band[0]
# write_image(img_3b, 'keypoint_3band_orig.tiff')

# Threshold (Otsu's Binarization)
ret, thresh = threshold(img.copy(), 0, 255, THRESH_OTSU)

# Erode foreground to seperate objects to define 'certainty area'
fg = erode(thresh, None, iterations=2)

# Dialate threshhold to define uncertainty area 'edge of the boat'
bgt = dilate(thresh, None, iterations=2)

# Threshold to define background
ret, bg = threshold(bgt, 1, 128, 1)

# Merge known foreground and known background images
marker = add(fg, bg)

# Label the region and run wathershed algorithm
marker32 = int32(marker)
watershed(img_3b, marker32)
watershed_img = convertScaleAbs(marker32)  # go back to 8 bit

# Cut out foreground from original image
ret, thresh = threshold(watershed_img, 0, 255, THRESH_BINARY+THRESH_OTSU)
foreground = bitwise_and(img, img, mask=thresh)
# write_image(foreground, 'keypoint_foreground.tif')
# plot_image(foreground)

#
# Technique 2 -  marker-based image segmentation using watershed algorithm
#                (uses distance transform)
img = imread('keypoint_1band_orig.tif')
gray = cvtColor(img, COLOR_BGR2GRAY)
ret, thresh = threshold(gray, 0, 255, THRESH_BINARY_INV+THRESH_OTSU)

# noise removal
kernel = ones((3, 3), uint8)
opening = morphologyEx(thresh, MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = distanceTransform(opening, DIST_L2, 5)
ret, sure_fg = threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = uint8(sure_fg)
unknown = subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
watershed(img, markers)
plot_image(img)
img[markers == -1] = [255, 0, 0]
