import cv2
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle
import subprocess
import time

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
PATHS = ['data/' + structure + '/' for structure in STRUCTURES]

def resize_all(path, new_size=(400,300)):
  try:
    os.mkdir(path + "sd/")
  except OSError:
    print "os error"
  for i,image in enumerate(os.listdir(path)):
    if image[len(image)-4:len(image)] != '.png':
      continue
    full_name = path + image
    im = cv2.imread(full_name)
    resized_im = cv2.resize(im, new_size)
    print 'writing out to', path+'sd/'+image
    cv2.imwrite(path + 'sd/' + image, resized_im)

for path in PATHS:
  resize_all(path)