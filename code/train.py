import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
from model import DRDCNN,FRDCNN,CRDCNN
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
import numpy as np
from keras import backend as K
from keras.models import Sequential, Model, model_from_json 
from keras.layers import (
    Input,
    Activation,
    concatenate,
    merge,
    Dropout,
    Reshape,
    Permute,
    Dense,
    UpSampling2D,
    Flatten
    )
from keras.optimizers import SGD, RMSprop
from keras.layers.convolutional import (
    Convolution2D)
from keras.layers.pooling import (
    MaxPooling2D,
    AveragePooling2D
    )
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
 

img_path_unet = '/home/qian/Desktop/projects/Organoid/CRDCNN_CellCounting/data/Unet_image'
lab_path_unet = '/home/qian/Desktop/projects/Organoid/CRDCNN_CellCounting/data/Unet_label'
img_path_FM = '/home/qian/Desktop/projects/Organoid/CRDCNN_CellCounting/data/FM_image'
lab_path_FM = '/home/qian/Desktop/projects/Organoid/CRDCNN_CellCounting/data/FM_label'
path1 = '/home/qian/Desktop/projects/Organoid/CRDCNN_CellCounting/data/real_data/AKTP'
path2 = '/home/qian/Desktop/projects/Organoid/CRDCNN_CellCounting/data/real_data/AKTP_Nt5e'
path3 = '/home/qian/Desktop/projects/Organoid/CRDCNN_CellCounting/data/real_data/AKTP_P2rx7'

input_dim = (256,256,3)
weights_unet = 'Xie.hdf5'
weights_FM = 'FM.h5'
weights_DRDCNN = 'DRDCNN.h5'
weights_FRDCNN = 'FRDCNN.h5'
weights_CRDCNN = 'CRDCNN.h5'
FRDCNN_pre = 'FRDCNN.json'
CRDCNN_pre = 'CRDCNN.json'

def rgb2gray(rgb):
	return np.dot(rgb[...,:3],[0.299,0.587,0.114])

def read_data_unet(img_path,lab_path):
	data_Unet = []
	anno_Unet = []
	label_Unet = []

	imList = os.listdir(img_path)
	for i in range(len(imList)): 
		img = Image.open(os.path.join(img_path,imList[i]))
		img = np.asarray(img)
		data_Unet.append(img)
	imList = os.listdir(lab_path)
	for i in range(len(imList)): 
		img = Image.open(os.path.join(lab_path,imList[i]))
		img = np.asarray(img)
		img = 100.0 * (img[:,:,0] > 0)
		anno_Unet.append(img)
		label_Unet.append(np.sum(np.sum(img,1),0)/100.)
	data_Unet = np.asarray(data_Unet, dtype = 'float32')
	mean = np.mean(data_Unet)
	std = np.std(data_Unet)  
	data_Unet = (data_Unet - mean) / std 
	anno_Unet = np.asarray(anno_Unet, dtype = 'float32')
	anno_Unet = np.expand_dims(anno_Unet, axis = -1)
	label_Unet = np.asarray(label_Unet, dtype = 'float32')
	return data_Unet, anno_Unet, label_Unet

def read_data_FM(img_path,lab_path):
	data_FM = []
	anno_FM = []
	label_FM = []

	imList = os.listdir(img_path)
	imList.sort()
	for i in range(len(imList)): 
		img = Image.open(os.path.join(img_path,imList[i]))
		img = np.asarray(img.resize((256,256)))#keep size insistant with Unet data
		img_R = np.interp(img,(img.min(),img.max()),(0,10))
		img_G = np.interp(img,(img.min(),img.max()),(0,10))
		img_B = np.interp(img,(img.min(),img.max()),(0,255))
		img = cv2.merge((img_R,img_G,img_B))#gnerate RGB chennels
		data_FM.append(img)

	imList = os.listdir(lab_path)
	imList.sort()
	for i in range(len(imList)): 
		img = Image.open(os.path.join(lab_path,imList[i]))
		img = np.asarray(img.resize((256,256)))
		img = 100.0 * (img > 0)
		anno_FM.append(img)
		label_FM.append(imList[i].split('_')[2][1:])

	data_FM=np.asarray(data_FM, dtype = 'float32')
	mean = np.mean(data_FM)
	std = np.std(data_FM)  
	data_FM = (data_FM - mean) / std  
	anno_FM = np.asarray(anno_FM, dtype = 'float32')
	anno_FM = np.expand_dims(anno_FM, axis = -1)
	label_FM = np.asarray(label_FM, dtype = 'float32')
	return data_FM, anno_FM, label_FM

def data_orgnization(img_path_unet,lab_path_unet,img_path_FM,lab_path_FM):
	data_Unet, anno_Unet, label_Unet = read_data_unet(img_path_unet,lab_path_unet)
	data_FM, anno_FM, label_FM = read_data_FM(img_path_FM,lab_path_FM)

	idx_Unet = [i for i in range(len(data_Unet))]
	idx_FM = [i for i in range(len(data_FM))]
	
	seed(1)
	train_idx_Unet = sample(idx_Unet,150)
	train_data_Unet,train_anno_Unet = data_Unet[train_idx_Unet],anno_Unet[train_idx_Unet]
	train_counts_Unet = label_Unet[train_idx_Unet]
	val_idx_Unet = np.delete(idx_Unet,train_idx_Unet) 
	val_data_Unet,val_anno_Unet = data_Unet[val_idx_Unet],anno_Unet[val_idx_Unet]
	val_counts_Unet = label_Unet[val_idx_Unet]

	seed(2)
	train_idx_FM = sample(idx_FM,1150)
	train_data_FM,train_anno_FM = data_FM[train_idx_FM],anno_FM[train_idx_FM]#
	train_counts_FM = label_FM[train_idx_FM] 
	val_idx_FM = np.delete(idx_FM,train_idx_FM) 
	val_data_FM,val_anno_FM = data_FM[val_idx_FM],anno_FM[val_idx_FM]
	val_counts_FM = label_FM[val_idx_FM]

	#train 
	train_data_all = np.concatenate((train_data_Unet,train_data_FM),axis = 0)
	train_anno_all = np.concatenate((train_anno_Unet,train_anno_FM),axis = 0)
	train_counts_all = np.concatenate((train_counts_Unet,train_counts_FM),axis = 0) 
	#test  
	val_data_all = np.concatenate((val_data_Unet,val_data_FM),axis = 0)
	val_anno_all = np.concatenate((val_anno_Unet,val_anno_FM),axis = 0)
	val_counts_all = np.concatenate((val_counts_Unet,val_counts_FM),axis = 0)      
	return train_data_all, train_anno_all, train_counts_all, val_data_all, val_anno_all, val_counts_all

def read_realdata(path):
	data = []
	imList = os.listdir(path)
	imList.sort()
	for i in range(len(imList)): 
		img = Image.open(os.path.join(path,imList[i]))
		img = np.asarray(img.resize((256,256)))#keep size insistant with sythetic data
		img_R = np.interp(img,(img.min(),img.max()),(0,10))
		img_G = np.interp(img,(img.min(),img.max()),(0,10))
		img_B = np.interp(img,(img.min(),img.max()),(0,255))
		img = cv2.merge((img_R,img_G,img_B))#gnerate RGB chennels
		data.append(img)
	
	data = np.asarray(data,dtype='float32')
	mean = np.mean(data)
	std = np.std(data)  
	data = (data - mean) / std 
	return data

def train(img_path_unet,lab_path_unet,img_path_FM,lab_path_FM,input_dim,FRDCNN,CRDCNN,weights_unet,weights_FM,weights_DRDCNN,weights_FRDCNN,weights_CRDCNN,path1,path2,path3):
	train_data_all, train_anno_all, train_counts_all, val_data_all, val_anno_all, val_counts_all = data_orgnization(img_path_unet,lab_path_unet,img_path_FM,lab_path_FM)
	AKTP = read_realdata(path1)
	AKTP_Nt5e = read_realdata(path2)
	AKTP_P2rx7 = read_realdata(path3)

	DRDCNN_model = DRDCNN(input_dim,weights_unet)
	#DRDCNN.fit(train_data_all,train_counts_all,batch_size=3,epochs = 3000,validation_split=0.1)
	DRDCNN_model.load_weights(weights_DRDCNN)
	DRDCNN_predict = DRDCNN_model.predict(val_data_all)
	DRDCNN_predict_AKTP = DRDCNN_model.predict(AKTP)
	DRDCNN_predict_AKTP_Nt5e = DRDCNN_model.predict(AKTP_Nt5e)
	DRDCNN_predict_AKTP_P2rx7 = DRDCNN_model.predict(AKTP_P2rx7)
	
	json_file = open(FRDCNN, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	FRDCNN_model = model_from_json(loaded_model_json)
	#FRDCNN_model = FRDCNN(model,weights_FM)
	#FRDCNN.fit(train_data_all,train_counts_all,batch_size=3,epochs = 500,validation_split=0.1) 
	FRDCNN_model.load_weights(weights_FRDCNN)
	FRDCNN_predict = FRDCNN_model.predict(val_data_all)
	FRDCNN_predict_AKTP = FRDCNN_model.predict(AKTP)
	FRDCNN_predict_AKTP_Nt5e = FRDCNN_model.predict(AKTP_Nt5e)
	FRDCNN_predict_AKTP_P2rx7 = FRDCNN_model.predict(AKTP_P2rx7)
	json_file = open(CRDCNN, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	CRDCNN_model = model_from_json(loaded_model_json)
	#CRDCNN_model = CRDCNN(input_dim,model,weights_unet,weights_FM)
	#CRDCNN.fit([train_data_all,train_data_all],train_counts_all,batch_size=4,epochs = 2500,validation_split=0.1) 
	CRDCNN_model.load_weights(weights_CRDCNN)
	CRDCNN_predict = CRDCNN_model.predict([val_data_all,val_data_all])
	CRDCNN_predict_AKTP = CRDCNN_model.predict([AKTP,AKTP])
	CRDCNN_predict_AKTP_Nt5e = CRDCNN_model.predict([AKTP_Nt5e,AKTP_Nt5e])
	CRDCNN_predict_AKTP_P2rx7 = CRDCNN_model.predict([AKTP_P2rx7,AKTP_P2rx7])
	
	resultFile = open("predict_val.csv",'w',newline="")
	wr = csv.writer(resultFile)
	wr.writerows((val_counts_all,DRDCNN_predict,FRDCNN_predict,CRDCNN_predict))

	resultFile = open("predict_AKTP.csv",'w',newline="")
	wr = csv.writer(resultFile)
	wr.writerows((DRDCNN_predict_AKTP,FRDCNN_predict_AKTP,CRDCNN_predict_AKTP))

	resultFile = open("predict_AKTP_Nt5e.csv",'w',newline="")
	wr = csv.writer(resultFile)
	wr.writerows((DRDCNN_predict_AKTP_Nt5e,FRDCNN_predict_AKTP_Nt5e,CRDCNN_predict_AKTP_Nt5e))
	
	resultFile = open("predict_AKTP_P2rx7.csv",'w',newline="")
	wr = csv.writer(resultFile)
	wr.writerows((DRDCNN_predict_AKTP_P2rx7,FRDCNN_predict_AKTP_P2rx7,CRDCNN_predict_AKTP_P2rx7))

os.chdir('/home/qian/Desktop/projects/Organoid/CRDCNN_CellCounting/result')

if __name__ == '__main__':
	train(img_path_unet,lab_path_unet,img_path_FM,lab_path_FM,input_dim,FRDCNN_pre,CRDCNN_pre,weights_unet,weights_FM,weights_DRDCNN,weights_FRDCNN,weights_CRDCNN,path1,path2,path3)

