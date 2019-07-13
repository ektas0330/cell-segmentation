import sys
sys.path.insert(0, 'C:\\Users\\SMARTS_Station\\Desktop\\CellSegmentation\\keras_version\\keras_version')
import os, cv2, fnmatch
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
K.set_image_data_format('channels_last') 
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, UpSampling2D, Dropout, Lambda, Activation, Add
from losses import iou_lovasz_multi,lovasz_softmax
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from normalized_optimizer_wrapper import NormalizedOptimizer
from utils import rotate_mirror_do, rotate_mirror_undo, windowed_subdivs, recreate_from_subdivs

def get_iou_vector(A, B):
    # Numpy version    
    batch_size = 1
    metric = 0.0
    for batch in range(batch_size):
        t, p = A, B
        true = np.sum(t)
        pred = np.sum(p)
        
        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            iouf = ((480*640)-pred)/(480*640)
        else:
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels
            intersection = np.sum(t * p)
            union = true + pred - intersection
            iouf = intersection / union # real iou
    return iouf

test_src_files = fnmatch.filter(os.listdir('./test/src/'), '*.jpg')
model = load_model('weights/unet_xception_resnet_nsgd32_lovaszsoftmax_best.h5',custom_objects={'lovasz_softmax': lovasz_softmax, 'iou_lovasz_multi': iou_lovasz_multi})


train_mean, train_std = np.load('train_mean_std.npy')
    
img_w, img_h = (256, 256) 
h, w = (480, 640)
overlap_pct = 0 
window_size = 256
n_w = (w-img_w*overlap_pct)//(img_w*(1-overlap_pct))+1
n_h = (h-img_h*overlap_pct)//(img_h*(1-overlap_pct))+1

aug_w = int((img_w*(1-overlap_pct)*n_w+img_w*overlap_pct-w)/2)
aug_h = int((img_h*(1-overlap_pct)*n_h+img_h*overlap_pct-h)/2)
borders = ((aug_h, aug_h), (aug_w, aug_w))

if not os.path.exists('./test/predict/'):
    os.makedirs('./test/predict/')

for filename in test_src_files:
    print(filename)
    img = cv2.imread('./test/src/' + filename, 0)
    pad = np.pad(img, pad_width=borders, mode='reflect')
    pads = rotate_mirror_do(pad)
    res = []
    padded_out_shape = list([512,768,3])
    for pad in tqdm(pads):
        # For every rotation:
        sd = windowed_subdivs(model, 3, train_mean, train_std, pad, window_size, overlap_pct)
        one_padded_result = recreate_from_subdivs(sd, window_size, overlap_pct, padded_out_shape)
        res.append(one_padded_result)
    # Merge after rotations:
    padded_results = rotate_mirror_undo(res)
    #convert the output of a model with lovasz loss as the metric to probability
    prob = np.exp(padded_results)/(1+np.exp(padded_results))
    prd = prob[aug_h:aug_h+h, aug_w:aug_w+w,:]
    label = np.load('./test/label/'+os.path.splitext(filename)[0]+'.npy')
    label = np.array(label, dtype=np.uint8)
    label_back = np.where(label > 0, 0 , 1)
    label_cell = np.where(label == 2, 1, 0)
    label_bead = np.where(label  ==1 , 1, 0)
    boolmulti = prd>0.51

    iouback = get_iou_vector(label_back,boolmulti[:,:,0])
    ioubead = get_iou_vector(label_bead,boolmulti[:,:,1])
    ioucell = get_iou_vector(label_cell,boolmulti[:,:,2])
    finaliou = (iouback+ioubead+ioucell)/3
    print(iouback,'tab', ioubead,'tab',ioucell)
    print(finaliou)
    cv2.imwrite('./test/predict/'+ os.path.splitext(filename)[0]+'.png', np.rint(prd*127))  
    
    
