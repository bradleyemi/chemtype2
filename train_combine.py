import cv2
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle
import random
from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
import subprocess
import time

LINE_WIDTH = 18
TEMPLATE_PATHS = [
'train/oh/',
'train/or/',
'train/o/',
'train/h/',
'train/n/',
'train/ro/'
]
SINGLE_TEMPLATE_PATHS = [
'train/o/',
'train/h/',
'train/n/',
'train/r/'
]
STRUCTURES = [
  'struct1', 
  'struct4', 
  'struct5',
  'struct8',
  'struct13',
  'struct16',
  'struct19',
  'struct20',
  'struct22',
]
PATHS = ['data/' + structure + '/sd/' for structure in STRUCTURES]
TEMPLATE_NAMES = ['OH', 'OR', 'O', 'H', 'N', 'RO']
SINGLE_TEMPLATE_NAMES = ['O', 'H', 'N', 'R']

TRAIN_IMAGES = ['01.png', '09.png', '17.png', '25.png', '33.png']
PYRAMID_SIZES = range(20,70,10)
STEP = 5

def get_max_size(x_list, y_list):
  s = max([max(x_list), max(y_list)])
  return (s,s)

# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
def deskew(img, SZ):
  m = cv2.moments(img)
  if abs(m['mu02']) < 1e-2:
    return img.copy()
  skew = m['mu11']/m['mu02']
  M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
  img = cv2.warpAffine(img,M,(SZ, SZ))
  return img

def crop_and_resize_image(thresh_im, size_before_pad, pad):
  #plt.imshow(thresh_im, cmap='Greys_r')
  #plt.title("input")
  #plt.show()
  height, width = thresh_im.shape
  top_crop = 0
  bottom_crop = 0
  left_crop = 0
  right_crop = 0
  # top
  for y in range(height):
    row = thresh_im[y,:]
    if np.count_nonzero(row) > 0:
      top_crop = y
      break
  for y in reversed(range(height)):
    row = thresh_im[y,:]
    if np.count_nonzero(row) > 0:
      bottom_crop = y
      break
  for x in range(width):
    col = thresh_im[:,x]
    if np.count_nonzero(col) > 0:
      left_crop = x
      break
  for x in reversed(range(width)):
    col = thresh_im[:,x]
    if np.count_nonzero(col) > 0:
      right_crop = x
      break
  #print top_crop, bottom_crop, left_crop, right_crop
  fully_cropped = thresh_im[top_crop:bottom_crop, left_crop:right_crop]
  #plt.imshow(fully_cropped, cmap='Greys_r')
  #plt.title("fully cropped")
  #plt.show()
  fully_cropped = cv2.resize(fully_cropped, (size_before_pad, size_before_pad))
  fully_cropped = cv2.copyMakeBorder(fully_cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
  #plt.imshow(square,cmap='Greys_r')
  #plt.title("output")
  #plt.show()
  return fully_cropped

def crop_and_make_templates(path):
  ims = []
  for i, image in enumerate(os.listdir(path)):
    if image[len(image)-4:len(image)] != '.png':
      continue
    if 'combined' in image:
      continue
    full_name = path+image
    im = cv2.imread(full_name,0)
    ret,im = cv2.threshold(im,100,255,cv2.THRESH_BINARY_INV)
    im = crop_and_resize_image(im,40,0)
    ims.append(im)
  return ims

def stack_templates(path, train_split = 0.9):
  ims = crop_and_make_templates(path)
  n_images = float(len(ims))
  ims = [cv2.GaussianFilter(im,(5,5),5) / n_images for im in ims]
  final_image = ims[0]
  for im in ims[1:]:
    final_image = cv2.add(final_image,im)
  cv2.imwrite(path + 'combined.png', final_image)
  plt.imshow(final_image,cmap='Greys_r')
  plt.show()

stack_templates('train/r/')
stack_templates('train/o/')
stack_templates('train/h/')
stack_templates('train/n/')

#crop_and_make_templates('train/r/')

'''
def stack_train_images(path):
  images = []
  heights = []
  widths = []
  for i,image in enumerate(os.listdir(path)):
    if image[len(image)-4:len(image)] != '.png':
      continue
    if 'combined' in image:
      continue
    full_name = path + image
    im = cv2.imread(full_name,0)
    height, width = im.shape
    ret,im = cv2.threshold(im,127,255,cv2.THRESH_BINARY_INV)
    im = cv2.GaussianBlur(im, (LINE_WIDTH/2, LINE_WIDTH/2), LINE_WIDTH/2)
    im = im / 5.0
    images.append(im)
    heights.append(height)
    widths.append(width)

  norm_images = []
  norm_size = get_max_size(heights, widths)
  s = norm_size[0]
  for im in images:
    y_size, x_size = im.shape
    y_pad = int((s-y_size) * 0.5)
    x_pad = int((s-x_size) * 0.5)
    norm_im = cv2.copyMakeBorder(im, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_CONSTANT, value=0)
    plt.imshow(norm_im, cmap='Greys_r')
    plt.title("Skewed")
    plt.show()
    norm_im = deskew(norm_im, norm_size[0])
    plt.imshow(norm_im, cmap='Greys_r')
    plt.title("Corrected")
    plt.show()
    norm_images.append(norm_im)

  final_image = norm_images[0]
  for im in norm_images[1:]:
    final_image = cv2.add(final_image, im)
  
  plt.imshow(final_image, cmap='Greys_r')
  plt.show()

  cv2.imwrite(path + "combined_deskew.png", final_image)


'''
#for path in TEMPLATE_PATHS:
#  stack_train_images(path)

