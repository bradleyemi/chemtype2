'''
Testing other supervised classifiers for text detection.
'''

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

TRAIN_IMAGES = ['01.png', '09.png', '17.png', '25.png', '33.png']
PYRAMID_SIZES = range(20,70,10)

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
    im = crop_and_resize_image(im,35,5)
    ims.append(im)
  return ims

def get_training_examples(path, n_per_size=5):
  for i, image in enumerate(os.listdir(path)):
    if image[len(image)-6:len(image)] not in TRAIN_IMAGES:
      continue
    print image
    im = cv2.imread(path+image,0)
    ret,im = cv2.threshold(im,100,255,cv2.THRESH_BINARY_INV)
    height,width = im.shape
    j = 0
    for size in PYRAMID_SIZES:
      print size
      n_subs = 0
      while n_subs < n_per_size:
        x = random.choice(range(0,width-size))
        y = random.choice(range(0,height-size))
        sub = im[y:y+size, x:x+size]
        if np.count_nonzero(sub) == 0:
            continue
        plt.imshow(sub, cmap='Greys_r')
        plt.ion()
        plt.show()
        letter = raw_input("Letter (x for none)--> ")
        plt.close()
        if letter == 'x':
          cv2.imwrite('train/none/' + image + '_' + str(j) + '.png', sub)
          n_subs += 1
          j += 1
        elif letter in ['h','o','r', 'n']:
          print letter
          cv2.imwrite('train/' + letter + '/' + image + '_' + str(j) + '.png', sub)
          n_subs += 1
          j += 1
        else:
          print "letter not recognized"

#for path in PATHS:
#  get_training_examples(path)

# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
def hog(img):
  bin_n = 16
  gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
  gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
  mag, ang = cv2.cartToPolar(gx, gy)
  bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
  bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
  mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
  hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
  hist = np.hstack(hists)     # hist is a 64 bit vector
  return hist

def train_ocr_classifier(train_split=0.9, classifier_type='nn'):
  X = []
  y = []
  for image in os.listdir('train/none/'):
    if image[len(image)-4:len(image)] != '.png':
      continue
    im = cv2.imread('train/none/' + image,0)
    im = cv2.resize(im,(40,40))
    # Convolution layer
    im = cv2.GaussianBlur(im, (3,3), 3)
    # Pool layer
    pooled = np.zeros((20,20))
    for i in range(0,40,2):
      for j in range(0,40,2):
        pooled[i/2, j/2] = max([im[i,j], im[i+1,j], im[i,j+1], im[i+1,j+1]])
    # Second convolution layer
    pooled = cv2.GaussianBlur(im, (3,3), 3)
    # Second pooling layer
    pooled2 = np.zeros((10,10))
    for i in range(0,20,2):
      for j in range(0,20,2):
        pooled[i/2,j/2] = max([im[i,j], im[i+1,j], im[i,j+1], im[i+1,j+1]])
    # Take the second pooling layer as features
    features = list(np.reshape(pooled2,(1,100))[0])
    #features = hog(im)
    X.append(features)
    y.append(0)
  dirs = ['h', 'o', 'n', 'r']
  for n, path in enumerate(dirs):
    for im in crop_and_make_templates('train/' + path + '/'):
      # Convolution layer
      im = cv2.GaussianBlur(im, (3,3), 3)
      # Pool layer
      pooled = np.zeros((20,20))
      for i in range(0,40,2):
        for j in range(0,40,2):
          pooled[i/2, j/2] = max([im[i,j], im[i+1,j], im[i,j+1], im[i+1,j+1]])
      # Second convolution layer
      pooled = cv2.GaussianBlur(im, (3,3), 3)
      # Second pooling layer
      pooled2 = np.zeros((10,10))
      for i in range(0,20,2):
        for j in range(0,20,2):
          pooled[i/2,j/2] = max([im[i,j], im[i+1,j], im[i,j+1], im[i+1,j+1]])
      # Take the second pooling layer as features
      features = list(np.reshape(pooled2,(1,100))[0])
      #plt.imshow(im, cmap='Greys_r')
      #plt.show()
      #features = hog(im)
      X.append(features)
      y.append(n+1)

  n_examples = len(y)
  n_train = int(train_split*n_examples)
  train_indices = np.random.choice(range(n_examples), size=n_train, replace=False)
  X_train = []
  X_test = []
  weights_train = []
  y_train = []
  y_test = []
  for i in range(n_examples):
    if i in train_indices:
      X_train.append(X[i])
      y_train.append(y[i])
      if y[i] != 0:
        weights_train.append(10.0)
      else:
        weights_train.append(1.0)
    else:
      X_test.append(X[i])
      y_test.append(y[i])
  #classifier = LogisticRegression()
  #classifier = LinearSVC()
  #classifier = SVC()
  if classifier_type == 'nn':
    classifier = cv2.ANN_MLP(np.array([100,30,5]))
    y_train_array = np.zeros((len(y_train), 5))
    for i,y in enumerate(y_train):
      y_train_array[i,y] = 1.0
    classifier.train(np.array(X_train), y_train_array, np.array(weights_train))
    if train_split < 1:
      ret, output = classifier.predict(np.array(X_test))
      n_correct = 0.0
      n_total = 0.0
      for i, prediction in enumerate(output):
        n_total += 1
        if y_test[i] == np.argmax(prediction):
          n_correct += 1
      score = n_correct/n_total
      print score
    return classifier
  else:
    if classifier_type == 'svm':
      classifier = LinearSVC()
    if classifier_type == 'logistic_regression':
      classifier = LogisticRegression()
    classifier.fit(X_train,y_train)
    if train_split < 1:
      score = classifier.score(X_test, y_test)
      print score
    return classifier, classifier_type
'''
avg = []
for i in range(10):
  avg.append(train_ocr_classifier())
print np.mean(avg)
'''

def find_text(img, classifier_type='nn'):
  im = cv2.imread(img,0)
  ret,im = cv2.threshold(im,100,255,cv2.THRESH_BINARY_INV)
  im = cv2.GaussianBlur(im,(5,5),5)
  height, width = im.shape
  classifier, classifier_type = train_ocr_classifier(train_split=1, classifier_type=classifier_type)
  display_image = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
  for size in PYRAMID_SIZES:
    for y in range(0,im.shape[0]-size,STEP):
      for x in range(0,im.shape[1]-size,STEP):
        sub = im[y:y+size,x:x+size]
        if np.count_nonzero(sub) == 0:
          continue
        sub = cv2.resize(sub,(40,40))
        features = hog(sub)
        if classifier_type == 'nn':
          ret, letter_prob = classifier.predict(np.array([features]))
          letter = np.argmax(letter_prob)
          if letter == 0:
            continue
          else:
            ratio = float(letter_prob[0][letter]) / letter_prob[0][0]
            if ratio < 2.0:
              continue
        else:
          letter = classifier.predict([features])[0]
          if letter == 0:
            continue
        if letter == 1:
          color = [255,0,0]
        if letter == 2:
          color = [0,255,0]
        if letter == 3:
          color = [0,0,255]
        if letter == 4:
          color = [0,255,255]
        cv2.rectangle(display_image,(x,y),(x+size, y+size),color,2)
  plt.imshow(display_image)
  plt.show()
        
find_text('data/struct19/sd/struct19_01.png')

