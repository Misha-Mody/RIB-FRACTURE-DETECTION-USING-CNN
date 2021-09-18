import numpy as np
import os
import skimage, os
from skimage.util import montage
import matplotlib.pyplot as plt 
import math
from sklearn.model_selection import train_test_split
import tensorflow as tf 
def plot_ct_scan(scan):
    f, ax1 = plt.subplots(1,1, figsize=(12, 12))
    ax1.imshow(montage(scan), cmap=plt.cm.bone) 
    ax1.axis('off')

def normalize(img, norm_type='intensity'):

      if norm_type == 'intensity':
          return (img-img.min())/(img.max()-img.min())

      elif norm_type == 'zscore':
          mu = img[img.nonzero()].mean()
          sigma = img[img.nonzero()].std()
          return (img - mu)/(sigma)
          
      elif norm_type == 'hist-eq':
          return hist_equalize(img)
      else:
          return img

def get_orig_data(visualize):
  x= np.load("/content/gdrive/MyDrive/Kaggle/data/0-420X.npy")
  y=np.load('/content/gdrive/MyDrive/Kaggle/data/0-420Y.npy')
  
  print(f'Shape of x original is {x.shape} and y original is {y.shape}') 

  for i in range(0,len(y)):
    if np.amin(y[i]) == 0 and  np.amax(y[i])==0:
      continue
    y[i]= normalize(y[i])
    
  if visualize:
    plot_ct_scan(x[::40,:,:,0])
    plot_ct_scan(y[::40,:,:,0])
    
  return x,y

def train_test_val_split(x,y,testsplit):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsplit)
  print(f'Shape of  train is {x_train.shape} and test is {x_test.shape}')
  #x_train, x_val, y_train, y_val = train_test_split(x_train_1, y_train_1, test_size=valsplit)
  
  return x_train,y_train,x_test,y_test

def get_val():
  x_val =np.load("/content/gdrive/MyDrive/Kaggle/data/420-500X.npy")
  y_val =np.load('/content/gdrive/MyDrive/Kaggle/data/420-500Y.npy')
  print(f'Shape of val is {x_val.shape}')
  return x_val,y_val  

def get_aug(x_train,y_train,batchsize):  
  data_gen_args = dict(horizontal_flip=True)
  image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
  mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
  # Provide the same seed and keyword arguments to the fit and flow methods
  image_datagen.fit(x_train, augment=True, seed=1)
  mask_datagen.fit(y_train, augment=True, seed=1)
  image_generator = image_datagen.flow(x_train,batch_size=batchsize,seed=1)
  mask_generator = mask_datagen.flow(y_train,batch_size=batchsize,seed=1)
  # combine generators into one which yields image and masks
  train_generator = zip(image_generator, mask_generator)
  print(f'AUGMENTATION PARAM: {data_gen_args}')
  return train_generator,data_gen_args
