import cv2
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle
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
TEMPLATE_NAMES = ['OH', 'OR', 'O', 'H', 'N', 'RO']

def get_max_size(x_list, y_list):
  s = max([max(x_list), max(y_list)])
  return (s,s)

def stack_train_images(path):
  images = []
  heights = []
  widths = []

  for i,image in enumerate(os.listdir(path)):
    print i
    if image[len(image)-4:len(image)] != '.png':
      continue
    full_name = path + image
    im = cv2.imread(full_name,0)
    ret,im = cv2.threshold(im,127,255,cv2.THRESH_BINARY_INV)
    im = cv2.GaussianBlur(im, (LINE_WIDTH/2, LINE_WIDTH/2), LINE_WIDTH/2)
    im = im / 5.0
    height, width = im.shape
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
    norm_images.append(norm_im)

  final_image = norm_images[0]
  for im in norm_images[1:]:
    final_image = cv2.add(final_image, im)
  
  plt.imshow(final_image, cmap='Greys_r')
  plt.show()

  cv2.imwrite(path + "combined.png", final_image)

for path in TEMPLATE_PATHS:
  stack_train_images(path)

