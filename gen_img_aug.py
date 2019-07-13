import os, cv2, fnmatch, random
import numpy as np
from shutil import copy2
from sklearn.model_selection import train_test_split
np.random.seed(2018)

img_w, img_h = (256, 256) 
w, h = (640, 480)
folder = './augmented_75k/'

if not os.path.exists(folder + './train/src/'):
    os.makedirs(folder + './train/src/')

if not os.path.exists(folder + './test/src/'):
    os.makedirs(folder + './test/src/')

if not os.path.exists(folder + './test/label/'):
    os.makedirs(folder + './test/label/')

if not os.path.exists(folder + './train/label/'):
    os.makedirs(folder + './train/label/')

def chooseScale(src, label, randTemp):
    if randTemp < 1/2:
        #scale in
        begin_w = random.randint(0, w-2*img_w-1)
        begin_h = random.randint(0, h-2*img_h-1)
        src_sub = src[begin_h: begin_h + 2*img_h, begin_w: begin_w+ 2*img_w]
        label_sub = label[begin_h: begin_h + 2*img_h, begin_w: begin_w+ 2*img_w]
        src_sub_l1, label_sub_l1 = scale(src_sub, label_sub, 0.5) 
    elif 1/3 <= randTemp < 2/3:
        #scale out
        begin_w = random.randint(0, w-0.5*img_w-1)
        begin_h = random.randint(0, h-0.5*img_h-1)
        src_sub = src[begin_h: int(begin_h + 0.5*img_h), begin_w: int(begin_w+ 0.5*img_w)]
        label_sub = label[begin_h: int(begin_h + 0.5*img_h), begin_w: int(begin_w+0.5*img_w)]
        src_sub_l1, label_sub_l1 = scale(src_sub, label_sub, 2)
    else:           
        #no scale
        begin_w = random.randint(0, w-img_w-1)
        begin_h = random.randint(0, h-img_h-1)
        src_sub_l1 = src[begin_h: begin_h + img_h, begin_w: begin_w+img_w]
        label_sub_l1 = label[begin_h: begin_h + img_h, begin_w: begin_w+img_w]

    return src_sub_l1, label_sub_l1
    
def chooseRotate(src_sub_l1, label_sub_l1, randTemp):
    
    src_sub_r90 = np.rot90(src_sub_l1)
    label_sub_r90 = np.rot90(label_sub_l1)
    if randTemp < 1/8:
        src_sub_l2 = src_sub_l1
        label_sub_l2 = label_sub_l1
    elif 1/8 <= randTemp < 2/8:
        # rotate 90 degrees
        src_sub_l2 = src_sub_r90
        label_sub_l2 = label_sub_r90
    elif 2/8 <= randTemp < 3/8:
        # rotate 180 degrees
        src_sub_l2 = np.rot90(src_sub_l1, 2)
        label_sub_l2 = np.rot90(label_sub_l1, 2)      
    elif 3/8 <= randTemp < 4/8:
        # rotate 270 degrees
        src_sub_l2 = np.rot90(src_sub_l1, 3)
        label_sub_l2 = np.rot90(label_sub_l1, 3)      
    elif 4/8 <= randTemp < 5/8:
        # flip image horizontally
        src_sub_l2 = cv2.flip(src_sub_l1, 0)
        label_sub_l2 = cv2.flip(label_sub_l1, 0)    
    elif 5/8 <= randTemp < 6/8:
        # flip image vertically
        src_sub_l2 = cv2.flip(src_sub_l1, 1)
        label_sub_l2 = cv2.flip(label_sub_l1, 1)      
    elif 6/8 <= randTemp < 7/8:
        # flip image horizontally
        src_sub_l2 = cv2.flip(src_sub_r90, 0)
        label_sub_l2 = cv2.flip(label_sub_r90, 0)   
    else:
        # flip image vertically
        src_sub_l2 = cv2.flip(src_sub_r90, 1)
        label_sub_l2 = cv2.flip(label_sub_r90, 1)  

    return src_sub_l2, label_sub_l2

    

image_files = fnmatch.filter(os.listdir('./data/src'), '*.jpg') 
train_set, test_set = train_test_split(image_files, test_size=0.1, random_state=2018)  

for filename in test_set:
    copy2('./data/src/'+filename, folder+'test/src')
    copy2('./data/label/'+os.path.splitext(filename)[0]+'.npy', folder+'test/label')


image_num = 75000
image_each = image_num / len(train_set)

for filename in train_set:
    count = 1
    f = os.path.splitext(filename)[0]
    src = cv2.imread('./data/src/'+filename, 0)
    label = np.load('./data/label/'+f+'.npy')
    label = np.array(label, dtype=np.uint8)
    while count < image_each:
        temp = np.random.random()
        #scaling
        # we do not perform anyscaling
        src_sub, label_sub = chooseScale(src, label, 1)
        #rotate/flip
        src_sub, label_sub = chooseRotate(src_sub, label_sub, temp)
        
        cv2.imwrite((folder+'train/src/'+f+'_%d.jpg' % count), src_sub)
        cv2.imwrite((folder+'train/label/'+f+'_%d.jpg' % count), label_sub*127)        
        count += 1 
        
        
        

