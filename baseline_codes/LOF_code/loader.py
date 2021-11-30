import os

import numpy as np

from keras import backend
# from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export
from PIL import Image
import _pickle as cPickle

def load_batch(fpath, label_key='labels'):
  ret_val=[]
  labels_lis=[]
  for filename in os.listdir(fpath):
    with open(os.path.join(fpath, filename), 'rb') as f:
      im = Image.open(f)
      im = im.resize((224,224))
      data = np.array(im)

      labels = label_key
      
      ret_val.append(data)
      labels_lis.append(labels)

  ret_val = np.array(ret_val)
  labels_lis = np.array(labels_lis)

  # print('ret_val',ret_val.shape)
  # print('label',labels_lis.shape)

  return ret_val,labels_lis


def load_data(classes ,train_dir,val_dir,class_names,num_train_samples,num_val_samples, label_rate=0):
 
  x_train = np.empty((num_train_samples, 224, 224 ,3), dtype='uint8')
  y_train = np.empty((num_train_samples,), dtype='uint8')
  noisy_idx=[]

  if(label_rate!=0):
    # print("hello")
    per_class = (num_train_samples//classes)
    for i in range(1, classes+1):
      fpath = os.path.join(train_dir, class_names[i-1])
      # (x_train[(i - 1) * per_class:i * per_class, :, :, :],
      #  y_train[(i - 1) * per_class:i * per_class]) = load_batch(fpath,i-1)
      x_tr, y_tr = load_batch(fpath,i-1)
      x_clean = x_tr[0:int(x_tr.shape[0]*(1-label_rate))]
      y_clean = y_tr[0:int(y_tr.shape[0]*(1-label_rate))].reshape(-1,1)
      # print(y_clean)
      x_noise = x_tr[0:int(x_tr.shape[0]*(label_rate))]
      #y_noise = np.repeat(1, int(x_tr.shape[0]*label_rate)).reshape(-1,1)
      y_noise = np.random.choice(3,int(x_tr.shape[0]*label_rate)).reshape(-1,1)
      # y_noise = np.squeeze(y_noise,axis=1)
      for idx,k in enumerate(y_noise):
        if(k==(i-1)):
          y_noise[idx] = (k+1)%3
      if(i==1):
        clean_x = x_clean
        # print('i=1',y_noise.shape)
        clean_y = y_clean
        noise_x = x_noise
        noise_y = y_noise

      else:
        # print('i=2',y_clean.shape)
        noise_y = noise_y.reshape(-1,1)
        clean_y = clean_y.reshape(-1,1)
        # print('i=2',clean_y.shape)

        clean_x = np.vstack((clean_x,x_clean))
        clean_y = np.vstack((clean_y,y_clean))
        noise_y = np.vstack((noise_y,y_noise))
        noise_x = np.vstack((noise_x,x_noise))

        # print('x',clean_x.shape)
        # print(noise_x.shape)
        # print('y',noise_y.shape)
        # print(clean_y.shape)

        # exit()
      np.random.shuffle(noise_y)
      # print(clean_x.shape)
      # print(noise_x.shape)
      noise_y = np.squeeze(noise_y,axis=1)
      clean_y = np.squeeze(clean_y,axis=1)

      x_train1 = np.concatenate((noise_x,clean_x),axis=0)
     
      # print(noise_y.shape)
      # print(clean_y.shape)
      y_train1 = np.concatenate((noise_y,clean_y),axis=0)
      print('X',x_train1.shape)
      print('Y',y_train1.shape)
  else:
    per_class = num_train_samples//classes
    for i in range(1, classes+1):
      fpath = os.path.join(train_dir, class_names[i-1])
      (x_train[(i - 1) * per_class:i * per_class, :, :, :],
       y_train[(i - 1) * per_class:i * per_class]) = load_batch(fpath,i-1)
    x_train1=x_train
    y_train1=y_train


  y_train = np.reshape(y_train1, (len(y_train1), 1))
  x_train = x_train1.transpose(0, 2, 1, 3)

  x_val = np.empty((num_val_samples, 224, 224 ,3), dtype='uint8')
  y_val = np.empty((num_val_samples,), dtype='uint8')

  if val_dir is not None:
    per_class = num_val_samples//classes
    for i in range(1, classes+1):
      fpath = os.path.join(val_dir, class_names[i-1])
      (x_val[(i - 1) * per_class:i * per_class, :, :, :],
       y_val[(i - 1) * per_class:i * per_class]) = load_batch(fpath,i-1) 

    y_val = np.reshape(y_val, (len(y_val), 1))
    x_val = x_val.transpose(0, 2, 1, 3)
  # exit()
  shuffler = np.random.permutation(len(x_train))

  x_train = x_train[shuffler]
  y_train = y_train[shuffler]

  # y_train = np.squeeze(y_train,axis=0)
  # y_val = np.squeeze(y_val,axis=0)
  return (x_train, y_train),(x_val,y_val)


if __name__=='__main__':

  class_names =['Benign','Invasive','Normal']
  num_train_samples = 225
  num_val_samples = 75
  (x_train, y_train),(x_val,y_val) = load_data(3,'./BACH_dataset/train','./BACH_dataset/validation',class_names,num_train_samples,num_val_samples)

  class_names1 =['InSitu']
  num_train_samples1 = 100
  num_val_samples1 = 0
  (x_train1, y_train1),(x_val1,y_val1) = load_data(1,'./BACH_dataset/noise',None,class_names1,num_train_samples1,num_val_samples1)


  # print(x_train.shape)
  # print(y_train.shape)

  # print(x_train1.shape)
  # print(y_train1.shape)