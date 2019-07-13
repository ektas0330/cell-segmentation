import numpy as np
import os, cv2, fnmatch, math
from utils import pre_process
    
train_src_files = fnmatch.filter(os.listdir('./train/src'), '*.jpg')
img_w, img_h = (256, 256) 

folder = './augmented_75k/'

if not os.path.exists('./train/prep/'):
    os.makedirs('./train/prep/')


class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0


    def variance(self):
        return self.new_s / self.n if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())

rs = RunningStats()

for f in train_src_files:    
    img_rgb = cv2.imread('./train/src/' + f, 0)
    rs.push(img_rgb)
    
mean = rs.mean()
variance = rs.variance()
train_mean = np.mean(mean)
train_std = math.sqrt(np.sum(variance + np.multiply(mean, mean))/(img_h*img_w)-train_mean**2)

np.save('./train_mean_std', np.array([train_mean, train_std]))    

for f in train_src_files:
    img = cv2.imread('./train/src/' + f, 0)
    img_prep = pre_process(img, train_mean, train_std)
    cv2.imwrite('./train/prep/'+f, img_prep)