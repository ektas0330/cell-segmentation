import sys
sys.path.insert(0, 'C:\\Users\\SMARTS_Station\\Desktop\\CellSegmentation\\keras_version\\keras_version')

import os, cv2, fnmatch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, UpSampling2D, Dropout, Lambda, Activation, Add, LeakyReLU, ZeroPadding2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.applications.xception import Xception
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Adam, SGD
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.utils.vis_utils import plot_model

from normalized_optimizer_wrapper import NormalizedOptimizer
from snapshot_update_per_epoch import SnapshotCallbackBuilder

from losses import iou_lovasz_multi,lovasz_softmax

np.random.seed(2018)
img_w, img_h = (256, 256) 

init_lr = 0.001 #0.01 for bcedice
accum_it = 4.
filepath ='./train/'  

train_files = fnmatch.filter(os.listdir('./train/src'), '*.jpg')

train_set, val_set = train_test_split(train_files, test_size=0.2, stratify=None, random_state=2018)

# data for training   
def generateData(batch_size,data=[]):  
    print('generateData...')
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for filename in data: 
            batch += 1 
            img = cv2.imread(filepath + 'prep/' + filename)
            img = img_to_array(img)/255  
            train_data.append(img)  
            label = cv2.imread(filepath + 'label/' + filename, cv2.IMREAD_GRAYSCALE)
            label = np.rint(label/127)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size==0: 
                print ('get enough bacth!\n')
                train_data = np.array(train_data)  
                train_label = np.array(train_label)  
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  
                
def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x

def build_model(start_neurons):
    
    backbone = Xception(input_shape=(img_h, img_w, 3), weights='imagenet', include_top=False)
    input = backbone.input

    conv4 = backbone.layers[121].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)
    
     # Middle
    convm = Conv2D(start_neurons*32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, start_neurons*32)
    convm = residual_block(convm, start_neurons*32)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons*16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.1)(uconv4)
    
    uconv4 = Conv2D(start_neurons*16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = residual_block(uconv4, start_neurons*16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)
    
    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons*8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[31].output
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(0.1)(uconv3)
    
    uconv3 = Conv2D(start_neurons*8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons*8)
    uconv3 = residual_block(uconv3, start_neurons*8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons*4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[21].output
    conv2 = ZeroPadding2D(((1,0),(1,0)))(conv2)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons*4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons*4)
    uconv2 = residual_block(uconv2, start_neurons*4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    
    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons*2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[11].output
    conv1 = ZeroPadding2D(((3,0),(3,0)))(conv1)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons*2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons*2)
    uconv1 = residual_block(uconv1, start_neurons*2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    # 128 -> 256
    uconv0 = Conv2DTranspose(start_neurons*1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons*1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons*1)
    uconv0 = residual_block(uconv0, start_neurons*1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
    uconv0 = Dropout(0.1/2)(uconv0)
    output_layer_noActi = Conv2D(3, (1,1), padding="same", activation=None)(uconv0)
    model = Model(inputs=input, outputs=output_layer_noActi)

    return model

model_prefix = 'unet_xception_resnet_nsgd32_lovaszsoftmax'
model = build_model(16) 
sgd = SGD(init_lr, momentum=0.9, nesterov=True)
sgd = NormalizedOptimizer(sgd, normalization='l2')

model.compile(loss=lovasz_softmax, optimizer=sgd, metrics=[iou_lovasz_multi])
model.summary()  
plot_model(model, to_file='model.png', show_shapes=True)
train_numb = len(train_set)  
valid_numb = len(val_set)  
print ("the number of train data is", train_numb)  
print ("the number of val data is", valid_numb)
n_epochs = 25
n_cycles = 1
BS = 8

snapshot = SnapshotCallbackBuilder(n_epochs, n_cycles, init_lr)
H = model.fit_generator(generator=generateData(BS,train_set), steps_per_epoch=train_numb//(BS*accum_it)*accum_it, epochs=n_epochs, verbose=1,  
                validation_data=generateData(BS,val_set), validation_steps=valid_numb//(BS*accum_it)*accum_it, callbacks=snapshot.get_callbacks(model_prefix=model_prefix))  

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = n_epochs
plt.plot(np.arange(1, N+1), H.history["loss"], label="training loss")
plt.plot(np.arange(1, N+1), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(1, N+1), H.history["acc"], label="training accuracy")
plt.plot(np.arange(1, N+1), H.history["val_acc"], label="validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="best")
plt.savefig(model_prefix+'.png', dpi=300)
plt.close()

plt.figure()
plt.plot(np.arange(1, N+1), H.history["loss"], label="training loss")
plt.plot(np.arange(1, N+1), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(1, N+1), H.history["iou"], label="training IoU")
plt.plot(np.arange(1, N+1), H.history["val_iou"], label="validation IoU")
plt.xlabel("Epoch")
plt.ylabel("Loss/IoU")
plt.legend(loc="best")
plt.savefig(model_prefix+'_iou.png', dpi=300)