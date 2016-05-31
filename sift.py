'''
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
    #plt.imshow(out)
    #plt.show()

    # Also return the image if you'd like a copy
    return out

def get_sift_keypoints_single(img):
  im = cv2.imread(img)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  sift = cv2.SIFT()
  kp, des = sift.detectAndCompute(im,None)
  return kp, des, im

def match_keypoints(train_kp, train_des, train_im, img, outname):
  kp, des, im = get_sift_keypoints_single(img)
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(train_des, des,k=2)
  good = []
  for m,n in matches:
    #if m.distance < 0.75*n.distance:
    good.append(m)
  img3 = drawMatches(train_im,train_kp,im,kp,good)
  cv2.imwrite(outname, img3)
  return matches

def match_training_image(train_img, train_name, path):
  kp, des, train_im = get_sift_keypoints_single(train_img)
  for image in os.listdir(path):
    if image[len(image)-4:len(image)] != '.png':
      continue
    full_name = path+image
    match_keypoints(kp, des, train_im, full_name, 'sift_' + train_name + '_' + image)
'''