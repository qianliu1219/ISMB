
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
from generator import ImageDataGenerator
from model import buildModel_U_net
from keras import backend as K
from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler
from scipy import misc
import scipy.ndimage as ndimage

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

base_path = '/home/qian/Desktop/cells/'
data = []
anno = []

def step_decay(epoch):
    step = 16
    num =  epoch // step 
    if num % 3 == 0:
        lrate = 1e-3
    elif num % 3 == 1:
        lrate = 1e-4
    else:
        lrate = 1e-5
        #lrate = initial_lrate * 1/(1 + decay * (epoch - num * step))
    print('Learning rate for epoch {} is {}.'.format(epoch+1, lrate))
    return np.float(lrate)
    
def read_data(base_path):
    imList = os.listdir(base_path)
    for i in range(len(imList)): 
        if 'cell' in imList[i]:
            img1 = misc.imread(os.path.join(base_path,imList[i]))
            data.append(img1)
            
            img2_ = misc.imread(os.path.join(base_path, imList[i][:3] + 'dots.png'))
            img2 = 100.0 * (img2_[:,:,0] > 0)
            img2 = ndimage.gaussian_filter(img2, sigma=(1, 1), order=0)
            anno.append(img2)
    return np.asarray(data, dtype = 'float32'), np.asarray(anno, dtype = 'float32')



###########################         read dataset1     #####################################
img_path_Unet = '/home/qian/Desktop/projects/Organoid/cell_couting/data/data_for_models_training/data_for_density_extraction_model/image'
lab_path_Unet = '/home/qian/Desktop/projects/Organoid/cell_couting/data/data_for_models_training/data_for_density_extraction_model/label'

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
img_path_FM = '/home/qian/Desktop/projects/Organoid/cell_couting/data/data_for_models_training/data_for_foreground_extraction_model/image'
lab_path_FM = '/home/qian/Desktop/projects/Organoid/cell_couting/data/data_for_models_training/data_for_foreground_extraction_model/label'


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
##############################   dataset combine   ############################################ 
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

###########################################################################




    

model = buildModel_U_net(input_dim = (256,256,3))

    
model.load_weights('Xie.hdf5')

predicted_dots_maps = model.predict(val_data_all)

preicted_counts = np.abs(np.sum(np.sum(predicted_dots_maps,1),1)/100.0)

resultFile = open("Xie_predict.csv",'w',newline="")
wr = csv.writer(resultFile)
wr.writerows((val_counts_all,preicted_counts))

resultFile = open("Xie_predict.csv",'w',newline="")
wr = csv.writer(resultFile)
wr.writerows((val_counts_all,preicted_counts))
