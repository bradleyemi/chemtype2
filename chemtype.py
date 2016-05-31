import cv2
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle
import subprocess
import time


### globals 

THRESH_VAL = 100
LINE_WIDTH = 18 # needs to be even
BORDER = 30
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
STRUCTURES = ['struct19']
PATHS = ['data/' + structure + '/sd/' for structure in STRUCTURES]
TEMPLATES = ['train/oh/combined.png', 'train/or/combined.png', 'train/o/combined.png', 'train/h/combined.png', \
'train/n/combined.png', 'train/ro/combined.png']
TEMPLATE_NAMES = ['OH', 'OR', 'O', 'H', 'N', 'RO']

### ocr ground truth import ###

GROUND_TRUTH_DICT = {}
f = open('ocr_groundtruth.txt')
for line in f.readlines():
  split_line = line.split()
  k = split_line[0]
  vals = split_line[1:]
  vals = [int(v) for v in vals]
  GROUND_TRUTH_DICT[k] = vals
f.close()

### end ocr ground truth import ###

### corner ground truth import ###
CORNER_TRUTH_DICT = {}
g = open('corners_groundtruth.txt')
for line in g.readlines():
  split_line = line.split()
  k = split_line[0]
  v = split_line[1]
  CORNER_TRUTH_DICT[k] = int(v)
g.close()

## end corner ground truth import

# box is (x0,x1,y0,y1)
def inside_box(center_x,center_y,box):
  return (center_x < box[1] and center_x > box[0] and center_y < box[3] and center_y > box[2])

def template_match(template, img, min_scale=0.4, max_scale=1.0, n_scales=5, threshold = 0.6):
  im = cv2.imread(img,0)
  ret, im = cv2.threshold(im, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
  im = cv2.copyMakeBorder(im,BORDER,BORDER,BORDER,BORDER,cv2.BORDER_CONSTANT,0)
  im = cv2.GaussianBlur(im, (LINE_WIDTH/2, LINE_WIDTH/2), LINE_WIDTH/2)
  tem = cv2.imread(template,0)
  boxes = []
  for i,scale in enumerate(np.linspace(min_scale,max_scale,n_scales)[::-1]):
    tem_rescaled = cv2.resize(tem, None, fx=scale, fy=scale)
    w,h = tem_rescaled.shape[::-1]
    res = cv2.matchTemplate(im,tem_rescaled,cv2.TM_CCOEFF_NORMED)
    #plt.imshow(res, cmap="Greys_r")
    #plt.show()
    loc = np.where(res >= threshold)
    #print loc
    for pt in zip(*loc[::-1]):
      try:
        score = res[pt[1], pt[0]]
      except IndexError:
        continue
      flag = 0
      x0 = pt[0]
      x1 = pt[0]+w
      y0 = pt[1]
      y1 = pt[1]+h
      center_x = pt[0]+w/2
      center_y = pt[1]+h/2
      deletions = []
      for i,box in enumerate(boxes):
        if inside_box(center_x,center_y,box) and box[4] > score:
          flag = 1
        if inside_box(center_x,center_y,box) and box[4] < score:
          deletions.append(i)
      if flag == 0:
        cv2.rectangle(im,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)
        boxes.append((x0,x1,y0,y1,score))
  boxes = [boxes[i] for i in range(len(boxes)) if i not in deletions]
  return boxes

def all_template_match(templates, template_names, img, tol=0.6, display=False):
  template_dict = {}
  all_boxes = []
  corresponding_templates = []
  for i,template in enumerate(templates):
    boxes = template_match(template, img, threshold=tol)
    all_boxes += boxes
    for j in range(len(boxes)):
      corresponding_templates.append(i)
    #template_dict[template_names[i]] = all_boxes
  keep = [1 for box in all_boxes]
  for i,box1 in enumerate(all_boxes):
    for j in range(i+1,len(all_boxes)):
      box2 = all_boxes[j]
      center1x = (box1[0]+box1[1])/2
      center1y = (box1[2]+box1[3])/2
      center2x = (box2[0]+box2[1])/2
      center2y = (box2[2]+box2[3])/2
      if inside_box(center1x,center1y,box2) or inside_box(center2x,center2y,box1):
        score1 = box1[4]
        score2 = box2[4]
        if score1 >= score2:
          keep[j] = 0
        else:
          keep[i] = 0
  for i, template in enumerate(templates):
    template_dict[template_names[i]] = [all_boxes[k] for k in range(len(all_boxes)) \
      if corresponding_templates[k] == i and keep[k] == 1]

  if display:
    im = cv2.imread(img,0)
    ret, im = cv2.threshold(im, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
    im = cv2.copyMakeBorder(im,BORDER,BORDER,BORDER,BORDER,cv2.BORDER_CONSTANT,0)
    im = cv2.GaussianBlur(im, (LINE_WIDTH/2, LINE_WIDTH/2), LINE_WIDTH/2)

    for key in template_dict.keys():
      if len(template_dict[key]) != 0:
        for box in template_dict[key]:
          cv2.rectangle(im,(box[0],box[2]),(box[1],box[3]),(255,0,0),2)
    plt.imshow(im)
    plt.show()
  return template_dict

def all_template_match_all_images(templates, template_names, path, tol=0.6,display=False):
  true_pos = 0
  false_pos = 0
  false_neg = 0
  for i,image in enumerate(os.listdir(path)):
    if image[len(image)-4:len(image)] != '.png':
      continue
    full_name = path + image
    template_dict = all_template_match(templates, template_names, full_name, tol=tol, display=display)
    comparison = [template_dict['OH'], template_dict['OR'], template_dict['O'], \
    template_dict['H'], template_dict['N'], template_dict['RO']]
    comparison = [len(c) for c in comparison]
    truth = GROUND_TRUTH_DICT[image[0:8]]
    with open(image[0:len(image)-4] + '_tol_eq_' + str(tol) + '_template_bb.pickle', 'wb') as handle:
      pickle.dump(template_dict, handle)
    for i in range(len(comparison)):
      if comparison[i] == truth[i]:
        true_pos += comparison[i]
      if comparison[i] > truth[i]:
        false_pos += comparison[i] - truth[i]
        true_pos += truth[i]
      if comparison[i] < truth[i]:
        false_neg += truth[i] - comparison[i]
        true_pos += comparison[i]
  if true_pos + false_pos > 0:
    precision = float(true_pos) / (float(true_pos) + float(false_pos))
  else:
    precision = 1.0
  if true_pos + false_neg > 0:
    recall = float(true_pos) / (float(true_pos) + float(false_neg))
  else:
    recall = 1.0
  
  return precision, recall, true_pos, false_pos, false_neg

### Get tolerance prec/recall ###

n_tolerances = 9

test_tolerances = np.linspace(0.5,0.9,n_tolerances)
'''
p_file = open('precisions_by_struct.txt', 'w')
r_file = open('recalls_by_struct.txt', 'w')
total_file = open('precision_recall_by_tolerance.txt', 'w')

n_tp = np.zeros(n_tolerances)
n_fp = np.zeros(n_tolerances)
n_fn = np.zeros(n_tolerances)
for path in PATHS:
  p_file.write(path + ' ')
  r_file.write(path + ' ')
  precisions = []
  recalls = []
  for i,tol in enumerate(test_tolerances):
    start = time.time()
    print "Processing", path, "at tolerance", tol
    precision, recall, tp, fp, fn = all_template_match_all_images(TEMPLATES, TEMPLATE_NAMES, path, tol=tol, display=False)
    precisions.append(precision)
    recalls.append(recall)
    n_tp[i] += tp
    n_fp[i] += fp
    n_fn[i] += fn
    p_file.write(str(precision) + ' ')
    r_file.write(str(recall) + ' ')
    print "took", time.time()-start, "s"
  p_file.write('\n')
  r_file.write('\n')
p_file.close()
r_file.close()

for i in range(n_tolerances):
  total_file.write(str(float(n_tp[i])/(n_tp[i]+n_fp[i])) + ' ')
  total_file.write(str(float(n_tp[i])/(n_tp[i]+n_fn[i])) + '\n')
'''
'''
with open('precision_recall_by_tolerance.txt', 'r') as f:
  lines = f.readlines()
  precisions = [float(line.split()[0]) for line in lines]
  recalls = [float(line.split()[1]) for line in lines]

plt.scatter(test_tolerances,precisions,color='blue',label='Precision')
plt.scatter(test_tolerances,recalls,color='red',label='Recall')
plt.title("Precision/Recall Curve, OCR")
plt.xlabel("Tolerance")
plt.legend()
plt.show()
'''

### End get tolerance prec/recall

def corner_detector(img, max_corners = 20, display=False, rect_w=6):
  max_rgb_val = 255
  im = cv2.imread(img,0)
  # threshold the image to make binary
  ret,thresh1 = cv2.threshold(im,THRESH_VAL,255,cv2.THRESH_BINARY_INV)
  thresh1 = cv2.copyMakeBorder(thresh1,BORDER,BORDER,BORDER,BORDER,cv2.BORDER_CONSTANT,0)
  #plt.imshow(thresh1, cmap='Greys_r')
  #plt.show()
  # apply Gaussian filter
  blur = cv2.GaussianBlur(thresh1,(LINE_WIDTH+1,LINE_WIDTH+1),LINE_WIDTH+1)
  #plt.imshow(blur, cmap='Greys_r')
  #plt.show()
  # detect corners
  corners = cv2.goodFeaturesToTrack(blur, 20, 0.0001, blur.shape[0] * 0.15, blockSize=40, useHarrisDetector=1, k=0.04)
  thresh1 = cv2.cvtColor(thresh1,cv2.COLOR_GRAY2RGB)
  if corners is None:
    return 0
  for corner in corners:
    corner_points = corner[0]
    corner_y = int(corner_points[0])
    corner_x = int(corner_points[1])
    cv2.rectangle(thresh1,(corner_y-rect_w/2,corner_x-rect_w/2),(corner_y+rect_w/2,corner_x+rect_w/2),color=[255,0,0],thickness=-1)
  if display:
    plt.imshow(thresh1)
    plt.show()
  return len(corners)

def corner_detector_all(path, display=False):
  true_pos = 0
  false_pos = 0
  false_neg = 0
  for i,image in enumerate(os.listdir(path)):
    print image
    if image[len(image)-4:len(image)] != '.png':
      continue
    full_name = path + image
    truth = CORNER_TRUTH_DICT[image[0:8]]
    actual = corner_detector(full_name, display=display)
    if actual == truth:
      true_pos += truth
    if actual > truth:
      false_pos += actual - truth
      true_pos += truth
    if actual < truth:
      false_neg += truth - actual
      true_pos += actual
  precision = float(true_pos) / (float(true_pos) + float(false_pos))
  recall = float(true_pos) / (float(true_pos) + float(false_neg))
  return precision, recall, true_pos, false_pos, false_neg

'''
p_file = open("corners_precision_by_struct.txt", 'w')
r_file = open("corners_recall_by_struct.txt", 'w')

n_tp = 0
n_fp = 0
n_fn = 0
precisions = []
recalls = []
for path in PATHS:
  p_file.write(path + ' ')
  r_file.write(path + ' ')
  start = time.time()
  print "Processing", path
  precision, recall, tp, fp, fn = corner_detector_all(path, display=True)
  precisions.append(precision)
  recalls.append(recall)
  n_tp += tp
  n_fp += fp
  n_fn += fn
  p_file.write(str(precision) + ' ')
  r_file.write(str(recall) + ' ')
  print "took", time.time()-start, "s"
  p_file.write('\n')
  r_file.write('\n')
p_file.close()
r_file.close()

print "Overall precision:", float(n_tp) / (n_tp+n_fp)
print "Overall recall:", float(n_tp) / (n_tp+n_fn)

'''

### Bond detection

## training images

# Downscale images (0.7,0.7), then pad the images to 40? px width and clip edges, use both image and its reflection

# For each rotation between -20 and 20 degrees,

# Find the number of windows (window width 20 px, step size 10 px) and make cutouts

# Apply Gaussian blur of 1/4 line width

# Get HOG features

# Feed these into an classifier for a particular orientation

## test images

# Rotate the image and cut out a bounding box of 40 px width by the bond length, clip edges by 20 px

# Use RANSAC to determine the orientation of the bond wrt to vertical axis (closest bin)

# Find the number of windows (window width 20px, step size 10 px) and make cutouts

# Apply Gaussian blur of 1/4 line width

# Get HOG features

# Use the classifier





