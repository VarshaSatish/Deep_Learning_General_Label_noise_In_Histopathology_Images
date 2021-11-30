import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(10)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import torch
import torch.nn as nn
random.seed(42)
import pandas as pd

path = "/home/sahyadri/noisy_labels/Supervised_SimCLR/BACH/Photos/Training"
classes = os.listdir(path)
classes.sort()  
#print(classes)
id_classes = [classes[0],classes[2],classes[3]]                                                         
od_classes = [classes[1]]
#print(id_classes)
#print(od_classes)

id_data_files = []

all_data_files = glob.glob(path+'/*/*')

id_data_files = [x for x in all_data_files if (x.split('/')[-2]==id_classes[0] or x.split('/')[-2]==id_classes[1] or x.split('/')[-2]==id_classes[2])]
od_data_files = [x for x in all_data_files if (x.split('/')[-2]==od_classes[0])]
id_data_labels = [x.split('/')[-2] for x in id_data_files]


train_data, val_data, train_label_clean, val_label = train_test_split(id_data_files, id_data_labels, test_size=0.25, stratify=id_data_labels)
train_data_with_labels = [[x,x.split('/')[-2],'id'] for x in train_data]
val_data_with_labels = [[x,x.split('/')[-2],'id'] for x in val_data]
all_data_with_labels = [[x,x.split('/')[-2],'id'] for x in all_data_files]

def noisy_data(no_ood_noise, no_label_noise): # Per-class values
    
    corrupted_train_data = []
    total_ood = no_ood_noise * len(id_classes)
    
    ood_data = random.choices(od_data_files, k = total_ood)
    corrupted_labels = id_classes * total_ood

    ood_data_with_labels = [[x,corrupted_labels[i],'ood'] for i, x in enumerate(ood_data)]
    corrupted_train_data.extend(ood_data_with_labels)
    
    for c in id_classes:
        class_wise_data = [x for x in train_data_with_labels if x[1]==c]
        label_noise_data = class_wise_data[:no_label_noise]
        clean_data = class_wise_data[no_label_noise:]
        corrupted_train_data.extend(clean_data)
        without_current_class = id_classes.copy()
        without_current_class.remove(c)
        length_noise_array = int(len(label_noise_data)/len(without_current_class))
        corrupted_labels = np.array(without_current_class*length_noise_array)
        temp_array = np.array(label_noise_data)
        
        temp_array[:,1] = corrupted_labels
        temp_array[:,2] = 'lnoise'
        label_noise_data = temp_array.tolist()
        corrupted_train_data.extend(label_noise_data)

    with open('BACH_train_file.txt', 'w') as f:
        for item in corrupted_train_data:
            f.write("%s\n" % item)

    random.shuffle(corrupted_train_data)
    return corrupted_train_data

class TrainDataset(Dataset):

    def __init__(self,no_ood_noise, no_label_noise, csvfile=None, transform=None):
        self.transform = transform
        self.train_data = noisy_data(no_ood_noise,no_label_noise)
        self.csv = csvfile
    
    def change_csv(self, filecsv):
        self.csv = filecsv

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.train_data[idx][0]
        if(self.csv):
            df = pd.read_csv(self.csv)
            score = df.loc[df["path"]==img_path]["score"].values[0]
        image = Image.open(img_path).convert("RGB")
        img_class = self.train_data[idx][1]
        img_type = self.train_data[idx][2]

        if self.transform:
            x_i = self.transform(image)
            x_j = self.transform(image)

        classes = {'Benign':0.0,'Invasive':1.0,'Normal':2.0,'InSitu':3.0}
        types = {'id': 1.0,'ood':2.0,'lnoise':3.0}
        if(self.csv):
            sample = {'x_i':x_i,'x_j':x_j,'label':classes[img_class], 'type': types[img_type], 'path':img_path, 'score':score}
        else:
            sample = {'x_i':x_i,'x_j':x_j,'label':classes[img_class], 'type': types[img_type], 'path':img_path}
        
        return sample

class ValDataset(Dataset):
    
    def __init__(self, transform=None):
        self.transform = transform
        self.val_file_paths = val_data_with_labels

    def __len__(self):
        return len(self.val_file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.val_file_paths[idx][0]
        image = Image.open(img_path).convert("RGB")
        img_class = self.val_file_paths[idx][1]
        img_type = self.val_file_paths[idx][2]

        if self.transform:
            x_i = self.transform(image)
            x_j = self.transform(image)
        
        classes = {'Benign':0.0,'Invasive':1.0,'Normal':2.0,'InSitu':3.0}
        types = {'id': 1.0,'ood':2.0,'lnoise':3.0}
        sample = {'x_i':x_i,'x_j':x_j,'label':classes[img_class], 'type': types[img_type], 'path':img_path}

        return sample

# train_dat = TrainDataset(no_ood_noise=15, no_label_noise=14)
# train_loader = DataLoader(train_dat, batch_size=8,shuffle=True, num_workers=1)

# val_dat = ValDataset()
# val_loader = DataLoader(val_dat, batch_size=8,shuffle=True, num_workers=1)

# print(len(train_dat))
# print(len(val_dat))

