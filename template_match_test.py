'''
A script to test the template matching OCR.
'''

import chemtype

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

PATHS = ['data/' + structure + '/sd/' for structure in STRUCTURES]
TEMPLATES = ['train/oh/combined.png', 'train/or/combined.png', \
'train/o/combined.png', 'train/h/combined.png', 'train/n/combined.png', 'train/ro/combined.png']
TEMPLATE_NAMES = ['OH', 'OR', 'O', 'H', 'N', 'RO']

BOND_PATHS = ['train/single/', 'train/double/', 'train/triple/', 'train/dashed/', 'train/wedge/']
BOND_NAMES = ['single', 'double', 'triple', 'dashed', 'wedge']
COLOR_DICT = {
  'single':[255,0,0],
  'double':[0,0,255],
  'triple':[0,255,0],
  'dashed':[255,165,0],
  'wedge':[128,0,128],
  'none':[0,0,0]
}
COLOR_DICT_OCR = {
  'OH':[255,0,0],
  'OR':[0,255,0],
  'O':[0,0,255],
  'H':[255,255,0],
  'N':[0,255,255],
  'RO':[255,0,255]
}


## import training images into a dict

BOND_TRAINING_DICT = defaultdict(list)
for i,path in enumerate(BOND_PATHS):
  for image in os.listdir(path):
    if image[len(image)-4:len(image)] != '.png':
      continue
    BOND_TRAINING_DICT[BOND_NAMES[i]].append(path + image)

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

n_tolerances = 9

test_tolerances = np.linspace(0.5,0.9,n_tolerances)

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
  accuracies = []
  for i,tol in enumerate(test_tolerances):
    start = time.time()
    print "Processing", path, "at tolerance", tol
    precision, recall, tp, fp, fn, acc = all_template_match_all_images(TEMPLATES, TEMPLATE_NAMES, path, tol=tol, display=False)
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

for path in ['data/struct22/sd/']:
  precision, recall, tp, fp, fn, acc = all_template_match_all_images(TEMPLATES, TEMPLATE_NAMES, path, tol=0.77, display=True)
  print precision, recall, acc