import cv2
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import os
from PIL import Image
import subprocess

TEST_IMAGE = 'data/struct19_01.png'

# Source: http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    plt.imshow(out)
    plt.show()

    # Also return the image if you'd like a copy
    return out

def find_ocr(image):
  word_boxes = tool.image_to_string(
    Image.open(image),
    lang="eng",
    builder=pyocr.builders.WordBoxBuilder()
  )
  boxes = []
  texts = []
  for w in word_boxes:
    ustr = w.get_unicode_string()
    text, digits = process_unicode_str(ustr)
    boxes.append(digits)
    texts.append(text)
  return boxes, texts

DATA_PATH = 'data/'

def get_sift_keypoints(train_img, img):
  train = cv2.imread(train_img)
  im = cv2.imread(img)
  train = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  sift = cv2.SIFT()
  kp1, des1 = sift.detectAndCompute(train, None)
  kp2, des2 = sift.detectAndCompute(im, None)
  # BFMatcher with default params
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2, k=2)

  # Apply ratio test
  good = []
  for m,n in matches:
    if m.distance < 0.75*n.distance:
      good.append(m)
  
  # cv2.drawMatchesKnn expects list of lists as matches.
  img3 = drawMatches(train,kp1,im,kp2,good)
  cv2.imwrite('test2.png', img3)


full_path = '/Users/bradleyemi/chemtype2/'
get_sift_keypoints(full_path + 'train/train_oh.png', full_path + 'data/struct13_04.png')


def get_bounding_boxes(path):
  for image in os.listdir(path):
    if image[len(image)-4:len(image)] != '.png':
      continue
    full_name = path + image
    im = cv2.imread(path + image)
    ret, im = cv2.threshold(im,100,255,cv2.THRESH_BINARY_INV)
    im = cv2.copyMakeBorder(im,300,300,300,300,cv2.BORDER_CONSTANT,value=[0,0,0])
    fig1 = plt.figure()
    plt.imshow(im)
    ax = plt.gca()
    cv2.imwrite('processed/' + image, im)
    tess_cmd = ['tesseract', 'processed/' + image, 'stdout', 'bazaar']
    subprocess.call(tess_cmd)
    '''
    for box in boxes:
      x1,y1,x2,y2 = box[0:4]
      ax.add_patch(Rectangle((x1,y1), x2-x1, y2-y1,fill=False,linewidth=3,color='r'))
    plt.show()
    '''





