import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
from model import buildModel_U_net,buildModel_FCRN_A,buildModel_FCRN_A_v2
from keras import backend as K
from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler
from scipy import misc
import scipy.ndimage as ndimage
from keras.preprocessing.image import ImageDataGenerator
import csv
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
from keras.applications.vgg19 import VGG19
from keras.layers import MaxPooling1D,Dense,Activation,MaxPooling2D,Conv2D,Input,UpSampling2D,Dropout,concatenate,Flatten
from keras.models import Model
import keras
import cv2
from keras.layers.core import Reshape
from keras.models import model_from_json
from keras import regularizers
from random import seed
from random import sample
from PIL import Image
from keras.utils import plot_model

def rgb2gray(rgb):
	return np.dot(rgb[...,:3],[0.299,0.587,0.114])



                         ##########################################
                         ##########################################
                         #######     Data preparation       #######
                         ##########################################
                         ##########################################


###########################         read dataset1     #####################################
img_path_Unet = '/home/qian/Desktop/projects/Organoid/ISMB/data/data_for_models_training/data_for_density_extraction_model/image'
lab_path_Unet = '/home/qian/Desktop/projects/Organoid/ISMB/data/data_for_models_training/data_for_density_extraction_model/label'

#data_Unet
data_Unet = []
imList = os.listdir(img_path_Unet)
#imList.sort()
for i in range(len(imList)): 
        img = Image.open(os.path.join(img_path_Unet,imList[i]))
        img = np.asarray(img)
        data_Unet.append(img)


data_Unet = np.asarray(data_Unet, dtype = 'float32')
#normalization
mean = np.mean(data_Unet)
std = np.std(data_Unet)  
data_Unet = (data_Unet - mean) / std  


#anno_Unet  #label_Unet
anno_Unet = []
label_Unet = []
imList = os.listdir(lab_path_Unet)
#imList.sort()
for i in range(len(imList)): 
        img = Image.open(os.path.join(lab_path_Unet,imList[i]))
        img = np.asarray(img)
        img = 100.0 * (img[:,:,0] > 0)
        anno_Unet.append(img)
        label_Unet.append(np.sum(np.sum(img,1),0)/100.)


anno_Unet = np.asarray(anno_Unet, dtype = 'float32')
anno_Unet = np.expand_dims(anno_Unet, axis = -1)
label_Unet = np.asarray(label_Unet, dtype = 'float32')

#split training and testing 
seed(1)
idx_Unet = [i for i in range(200)]
train_idx_Unet = sample(idx_Unet,150)
train_data_Unet,train_anno_Unet = data_Unet[train_idx_Unet],anno_Unet[train_idx_Unet]
train_counts_Unet = label_Unet[train_idx_Unet]
val_idx_Unet = np.delete(idx_Unet,train_idx_Unet) 
val_data_Unet,val_anno_Unet = data_Unet[val_idx_Unet],anno_Unet[val_idx_Unet]
val_counts_Unet = label_Unet[val_idx_Unet]



########################         read dataset2      ######################################
img_path_FM = '/home/qian/Desktop/projects/Organoid/ISMB/data/data_for_models_training/data_for_foreground_extraction_model/image'
lab_path_FM = '/home/qian/Desktop/projects/Organoid/ISMB/data/data_for_models_training/data_for_foreground_extraction_model/label'


#data_FM
data_FM = []
imList = os.listdir(img_path_FM)
imList.sort()
for i in range(len(imList)): 
        img = Image.open(os.path.join(img_path_FM,imList[i]))
        img = np.asarray(img.resize((256,256)))#keep size insistant with Unet data
        img_R = np.interp(img,(img.min(),img.max()),(0,10))
        img_G = np.interp(img,(img.min(),img.max()),(0,10))
        img_B = np.interp(img,(img.min(),img.max()),(0,255))
        img = cv2.merge((img_R,img_G,img_B))#gnerate RGB chennels
        data_FM.append(img)


data_FM=np.asarray(data_FM, dtype = 'float32')
#normalization
mean = np.mean(data_FM)
std = np.std(data_FM)  
data_FM = (data_FM - mean) / std  


#anno_FM  #label_FM
anno_FM = []
label_FM = []
imList = os.listdir(lab_path_FM)
imList.sort()
for i in range(len(imList)): 
        img = Image.open(os.path.join(lab_path_FM,imList[i]))
        img = np.asarray(img.resize((256,256)))
        img = 100.0 * (img > 0)
        anno_FM.append(img)
        label_FM.append(imList[i].split('_')[2][1:])

anno_FM = np.asarray(anno_FM, dtype = 'float32')
anno_FM = np.expand_dims(anno_FM, axis = -1)
label_FM = np.asarray(label_FM, dtype = 'float32')


#split training and testing 
seed(2)
idx_FM = [i for i in range(1200)]
train_idx_FM = sample(idx_FM,1150)
train_data_FM,train_anno_FM = data_FM[train_idx_FM],anno_FM[train_idx_FM]#
train_counts_FM = label_FM[train_idx_FM] 
val_idx_FM = np.delete(idx_FM,train_idx_FM) 
val_data_FM,val_anno_FM = data_FM[val_idx_FM],anno_FM[val_idx_FM]
val_counts_FM = label_FM[val_idx_FM]




                        #                               #
                      #   #                           #   #
                     #     #                         #     #
##############################   dataset conbine   ############################################ 
#train_                                                                                       #
train_data_all = np.concatenate((train_data_Unet,train_data_FM),axis = 0)                     #
#train_anno_all = np.concatenate((train_anno_Unet,train_anno_FM),axis = 0)                    #
train_counts_all = np.concatenate((train_counts_Unet,train_counts_FM),axis = 0)               #
#test                                                                                         #
val_data_all = np.concatenate((val_data_Unet,val_data_FM),axis = 0)                           #
#val_anno_all = np.concatenate((val_anno_Unet,val_anno_FM),axis = 0)                          #
val_counts_all = np.concatenate((val_counts_Unet,val_counts_FM),axis = 0)                     #
###############################################################################################
                                #   #       #   #
                                #   #       #   #
                             ###    #        #   ###
                            #      #          #     #   
                             ######            #####

# save counts for all 1400 images
resultFile = open("counts_2datasets.csv",'w',newline="")
wr = csv.writer(resultFile)
wr.writerows([val_counts_all])

resultFile = open("counts_2datasets.csv",'w',newline="")
wr = csv.writer(resultFile)
wr.writerows([val_counts_all])







#########################        real data      #######################################

AKTP_path = '/home/qian/Desktop/projects/Organoid/ISMB/data/real_data/AKTP'
AKTP_Nt5e_path = '/home/qian/Desktop/projects/Organoid/ISMB/data/real_data/AKTP_Nt5e'
AKTP_P2rx7_path = '/home/qian/Desktop/projects/Organoid/ISMB/data/real_data/AKTP_P2rx7'

AKTP, AKTP_Nt5e,AKTP_P2rx7 = [],[],[]

##AKTP
imList = os.listdir(AKTP_path)
imList.sort()
for i in range(len(imList)): 
    img = Image.open(os.path.join(AKTP_path,imList[i]))
    img = np.asarray(img.resize((256,256)))#keep size insistant with sythetic data
    img_R = np.interp(img,(img.min(),img.max()),(0,10))
    img_G = np.interp(img,(img.min(),img.max()),(0,10))
    img_B = np.interp(img,(img.min(),img.max()),(0,255))
    img = cv2.merge((img_R,img_G,img_B))#gnerate RGB chennels
    AKTP.append(img)

AKTP = np.asarray(AKTP,dtype='float32')
###normalization
mean = np.mean(AKTP)
std = np.std(AKTP)  
AKTP = (AKTP - mean) / std  

##AKTP_Nt5e
imList = os.listdir(AKTP_Nt5e_path)
imList.sort()
for i in range(len(imList)): 
    img = Image.open(os.path.join(AKTP_Nt5e_path,imList[i]))
    img = np.asarray(img.resize((256,256)))#keep size insistant with sythetic data
    img_R = np.interp(img,(img.min(),img.max()),(0,10))
    img_G = np.interp(img,(img.min(),img.max()),(0,10))
    img_B = np.interp(img,(img.min(),img.max()),(0,255))
    img = cv2.merge((img_R,img_G,img_B))#gnerate RGB chennels
    AKTP_Nt5e.append(img)

AKTP_Nt5e = np.asarray(AKTP_Nt5e,dtype='float32')
###normalization
mean = np.mean(AKTP_Nt5e)
std = np.std(AKTP_Nt5e)  
AKTP_Nt5e = (AKTP_Nt5e - mean) / std  


##AKTP_P2rx7
imList = os.listdir(AKTP_P2rx7_path)
imList.sort()
for i in range(len(imList)): 
    img = Image.open(os.path.join(AKTP_P2rx7_path,imList[i]))
    img = np.asarray(img.resize((256,256)))#keep size insistant with sythetic data
    img_R = np.interp(img,(img.min(),img.max()),(0,10))
    img_G = np.interp(img,(img.min(),img.max()),(0,10))
    img_B = np.interp(img,(img.min(),img.max()),(0,255))
    img = cv2.merge((img_R,img_G,img_B))#gnerate RGB chennels
    AKTP_P2rx7.append(img)

AKTP_P2rx7 = np.asarray(AKTP_P2rx7,dtype='float32')
###normalization
mean = np.mean(AKTP_P2rx7)
std = np.std(AKTP_P2rx7)  
AKTP_P2rx7 = (AKTP_P2rx7 - mean) / std  


#######################################################################################


                         ################################################
                         ################################################
                         #######     build and train models       #######
                         ################################################
                         ################################################




############################
######    DRDCNN      ######
############################


#build and train encoder
# unet_model = buildModel_U_net(input_dim = (256,256,3))
# change_lr = LearningRateScheduler(step_decay)
# datagen = ImageDataGenerator(featurewise_center = False,samplewise_center = False, 
#     featurewise_std_normalization = False, samplewise_std_normalization = False, 
#     zca_whitening = False, rotation_range = 30, width_shift_range = 0.3,
#     height_shift_range = 0.3, zoom_range = 0.3,shear_range = 0.,horizontal_flip = True, 
#     vertical_flip = True, fill_mode = 'constant')  
# #unet_model.fit_generator(datagen.flow(train_data_Unet,train_anno_Unet,batch_size=16),
# #    steps_per_epoch=len(train_data_Unet) / 16,epochs = 192) # you can train the model by yout self as well


# #save encoder
# model_json = unet_model.to_json()
# with open("unet.json", "w") as json_file:
#     json_file.write(model_json)
# #unet_model.save_weights("unet.h5")# serialize weights to HDF5  #cell_counting_old.hdf5 #unet.h5



########################################################################
#########################    load Unet   ###############################
########################################################################
json_file = open('unet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
unet_model = model_from_json(loaded_model_json)
unet_model.load_weights("Xie.hdf5")#weights from Xie et al. #unet.h5
########################################################################


#build and train regression count_model
x = unet_model.output
x = BatchNormalization(name="batch_normalization_14")(x)
x = Conv2D(64,[3,3], kernel_regularizer=regularizers.l2(0.0001), padding='same',name="block1_1")(x)
x = LeakyReLU(alpha=0.1,name="activation_18")(x)
x = MaxPooling2D(pool_size=(28 ,28),strides=4,name="max_pooling2d_1_1")(x)
x = BatchNormalization(name="batch_normalization_2_1")(x)
x = Conv2D(32,[3,3],kernel_regularizer=regularizers.l2(0.0001),  padding='same',name="block2_1")(x)
x = LeakyReLU(alpha=0.1,name="activation_2_1")(x)
x = MaxPooling2D(pool_size=(7, 7),strides=4,name="max_pooling2d_2_1")(x)
# x = BatchNormalization(name="batch_normalization_3_1")(x)
# x = Conv2D(16,[3,3],kernel_regularizer=regularizers.l2(0.0001),  padding='same',name="block3_1")(x)
# x = LeakyReLU(alpha=0.1,name="activation_3_1")(x)
# x = MaxPooling2D(pool_size=(7, 7),strides=4,name="max_pooling2d_3_1")(x)
x = BatchNormalization(name="batch_normalization_4_1")(x)
x = Dense(4096,kernel_regularizer=regularizers.l2(0.0001))(x)
x = LeakyReLU(alpha=0.1,name="activation_4_1")(x)
x = MaxPooling2D(pool_size=(2,2),strides=2,name="max_pooling2d_4_1")(x)
x = BatchNormalization(name="batch_normalization_5_1")(x)
x = Dense(32,kernel_regularizer=regularizers.l2(0.0001))(x)
x = LeakyReLU(alpha=0.1,name="activation_5_1")(x)
x = MaxPooling2D(pool_size=(2,2),strides=2,name="max_pooling2d_5_1")(x)
x = BatchNormalization(name="batch_normalization_6_1")(x)
x = Flatten()(x)
x = Dense(1,kernel_regularizer=regularizers.l2(0.0001),activation="relu")(x)

DRDCNN = Model(inputs=unet_model.input, outputs=x)
for layer in unet_model.layers:
    layer.trainable = False


DRDCNN.compile(loss='mean_squared_error', 
	optimizer=keras.optimizers.Adam(lr=0.1,decay=1e-3), 
	metrics=['accuracy'])
#DRDCNN.summary()
#plot_model(DRDCNN, to_file='DRDCNN_model.png')
DRDCNN.fit(train_data_all,train_counts_all,batch_size=3,epochs = 3000,validation_split=0.1) 
print(DRDCNN.predict(val_data_all[1:5])) 

#save regression count_model
model_json = DRDCNN.to_json()
with open("DRDCNN.json", "w") as json_file:
    json_file.write(model_json)


DRDCNN.save_weights("DRDCNN.h5")# serialize weights to HDF5



########################################################################
####################    load count_model1   ############################
########################################################################
json_file = open('DRDCNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
DRDCNN = model_from_json(loaded_model_json)
DRDCNN.load_weights("DRDCNN.h5")
########################################################################




                         ################################################
                         ################################################
                         #######       results postprocess        #######
                         ################################################
                         ################################################


##############################     features image plots       ####################################
# plt.imshow(rgb2gray(val_data_all[51]),cmap=plt.get_cmap('gray'))
# plt.axis('off')
# plt.show()
# plt.imshow(unet_model.predict(val_data_all)[51].reshape(256,256),cmap=plt.get_cmap('gray'))
# plt.axis('off')
# plt.show()
# plt.imshow(FM_model.predict(val_data_all)[51].reshape(256,256),cmap=plt.get_cmap('gray'))
# plt.axis('off')
# plt.show()
# plt.imshow(val_anno_all[51].reshape(256,256),cmap=plt.get_cmap('gray'))
# plt.axis('off')
# plt.show()
# val_counts_all[51]#10

#############################      counts predictions   #########################################
DRDCNN_predict = DRDCNN.predict(val_data_all) 

#val_counts_all

# array([159., 257., 174., 273.,  96.,  79., 142., 244., 124., 218., 112.,
#        136.,  97., 108., 183., 202., 198., 162., 168., 162., 235., 192.,
#        306., 178., 194., 260., 206., 137., 155., 257., 244., 103., 314.,
#        214.,  78., 179., 220., 110., 223., 119., 130., 181., 258., 196.,
#        224.,  98., 199., 254., 218., 200.,   1.,  10.,  10.,  14.,  14.,
#         14.,  14.,  14.,  23.,  23.,  23.,  23.,  27.,  27.,  31.,  31.,
#         35.,  35.,  40.,  48.,  48.,  53.,  57.,  57.,  57.,  61.,  61.,
#         61.,  66.,  66.,  70.,  74.,  74.,  74.,  78.,  78.,  78.,  78.,
#         83.,  83.,  83.,  87.,  87.,  91.,  91.,  96., 100., 100., 100.,
#        100.], dtype=float32)


resultFile = open("DRDCNN_predict.csv",'w',newline="")
wr = csv.writer(resultFile)
wr.writerows((val_counts_all,DRDCNN_predict))

resultFile = open("DRDCNN_predict.csv",'w',newline="")
wr = csv.writer(resultFile)
wr.writerows((val_counts_all,DRDCNN_predict))



############################      real data predictions     #####################################

def Normalization2(x):
    k=(6-(-0.5))/(np.amax(x)-np.amin(x))
    return k*(x-np.amin(x))

AKTP = Normalization2(AKTP)
AKTP_P2rx7 = Normalization2(AKTP_P2rx7)
AKTP_Nt5e = Normalization2(AKTP_Nt5e)




AKTP_DRDCNN_predict = DRDCNN.predict(AKTP) 


resultFile = open("AKTP_DRDCNN_predict.csv",'w',newline="")
wr = csv.writer(resultFile)
wr.writerows((AKTP_DRDCNN_predict))

resultFile = open("AKTP_DRDCNN_predict.csv",'w',newline="")
wr = csv.writer(resultFile)
wr.writerows((AKTP_DRDCNN_predict))

AKTP_P2rx7_DRDCNN_predict = DRDCNN.predict(AKTP_P2rx7) 


resultFile = open("AKTP_P2rx7_DRDCNN_predict.csv",'w',newline="")
wr = csv.writer(resultFile)
wr.writerows((AKTP_P2rx7_DRDCNN_predict))

resultFile = open("AKTP_P2rx7_DRDCNN_predict.csv",'w',newline="")
wr = csv.writer(resultFile)
wr.writerows((AKTP_P2rx7_DRDCNN_predict))


AKTP_Nt5e_DRDCNN_predict = DRDCNN.predict(AKTP_Nt5e) 



resultFile = open("AKTP_Nt5e_DRDCNN_predict.csv",'w',newline="")
wr = csv.writer(resultFile)
wr.writerows((AKTP_Nt5e_DRDCNN_predict))

resultFile = open("AKTP_Nt5e_DRDCNN_predict.csv",'w',newline="")
wr = csv.writer(resultFile)
wr.writerows((AKTP_Nt5e_DRDCNN_predict))



# plt.imshow(unet_model.predict(AKTP)[0].reshape(256,256))
# plt.axis('off')
# plt.show()
# plt.imshow(FM_model.predict(AKTP)[0].reshape(256,256))
# plt.axis('off')
# plt.show()
