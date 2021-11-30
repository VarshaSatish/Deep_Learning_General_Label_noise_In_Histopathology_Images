import numpy as np
import time
import os,sys,argparse,time,math,csv
from operator import add
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms,datasets
import tensorboard_logger as tb_logger
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet,CustomModel
from torch.utils.data import Dataset, DataLoader
from dataloader_bach import TrainDataset, ValDataset
from sklearn.neighbors import LocalOutlierFactor
from numpy.linalg import norm
import torchvision.models as models
from torchvision import datasets, models, transforms
import pandas as pd
import myTransforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

random_seed = 1 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EXPT_NO = 1
OPTIMIZER = "sgd"
learning_rate = 0.01
warmup_epochs  = 10
ood_noise_val = 20
l_noise_val = 18

imagenet_weights = True   #(if false, it takes the contrastive weights by default)
if(imagenet_weights):
    folder_name ="./baseline_imagenet/Expt_baseline_{}_{}_ood_{}_lnoise_{}".format(EXPT_NO,OPTIMIZER,ood_noise_val,l_noise_val)
else:
    folder_name ="./baseline_contrastive/Expt_baseline_{}_{}_ood_{}_lnoise_{}".format(EXPT_NO,OPTIMIZER,ood_noise_val,l_noise_val)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=51,
                        help='number of training epochs')
    
    parser.add_argument('--warmup_epochs', type=int, default=warmup_epochs,
                        help='number of warmup phase epochs')
    parser.add_argument('--cross', type=int, default=20,
                        help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=learning_rate,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='10, 20, 40',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='BACH',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                    help='using cosine annealing')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    opt = parser.parse_args()
    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
  
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    opt.model_path = folder_name + '_models_single_weigh'.format(opt.dataset)
    opt.tb_path = folder_name + '_tensorboard_single_weigh'.format(opt.dataset)
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    opt.model_name = 'Expt_{}_{}_{}_lr_{}_decay_{}_{}'.\
        format(EXPT_NO, opt.dataset, opt.model, opt.learning_rate, OPTIMIZER)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    opt.tb_folder = os.path.join(folder_name, opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
      os.makedirs(opt.tb_folder)
    opt.save_folder = os.path.join(folder_name, opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt

def set_model(opt):

    # enable synchronized Batch Normalization
    if opt.syncBN:
        pre_model = apex.parallel.convert_syncbn_model(pre_model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pre_model = CustomModel()

    if(imagenet_weights):
        trained_model = torch.load("./resnet18-5c106cde_image_net.pth", map_location=torch.device('cpu'))
        for name, param in trained_model.items():
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            if 'fc' not in name:
                pre_model.state_dict()['resnet.'+name].copy_(param)
            elif (name == 'model.resnet.fc.1.weight'):
                pre_model.state_dict()['resnet.fc1.weight'].copy_(param)
            elif (name == 'model.resnet.fc.1.bias'):
                pre_model.state_dict()['resnet.fc1.bias'].copy_(param)
            elif (name == 'model.resnet.fc.3.weight'):
                pre_model.state_dict()['resnet.fc3.weight'].copy_(param)
            elif (name == 'model.resnet.fc.3.bias'):
                pre_model.state_dict()['resnet.fc3.bias'].copy_(param)

    else:
        trained_model = torch.load("./tenpercent_resnet18.ckpt", map_location=torch.device('cpu'))
        for name , param in trained_model['state_dict'].items():
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            if 'fc' not in name:
                pre_model.state_dict()['resnet.'+name[13:]].copy_(param)
            elif (name == 'model.resnet.fc.1.weight'):
                pre_model.state_dict()['resnet.fc1.weight'].copy_(param)
            elif (name == 'model.resnet.fc.1.bias'):
                pre_model.state_dict()['resnet.fc1.bias'].copy_(param)
            elif (name == 'model.resnet.fc.3.weight'):
                pre_model.state_dict()['resnet.fc3.weight'].copy_(param)
            elif (name == 'model.resnet.fc.3.bias'):
                pre_model.state_dict()['resnet.fc3.bias'].copy_(param)

    model = pre_model.to(device)
    # criterion = criterion.to(device)
    cudnn.benchmark = True

    return model

def set_loader(opt):
    # construct data loader

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    composed_tf = transforms.Compose([myTransforms.RandomHorizontalFlip(),
                                        myTransforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1),
                                        myTransforms.RandomVerticalFlip(),
                                        myTransforms.Resize((224,224)),
                                        myTransforms.ToTensor(),
                                        myTransforms.Normalize(mean = mean, std = std)
                                      ])
    composed_notf = transforms.Compose([myTransforms.Resize((224,224)),
                                        myTransforms.ToTensor(),
                                        myTransforms.Normalize(mean = mean, std = std)
                                      ])

    train_dat=TrainDataset(transform=composed_tf,no_ood_noise=ood_noise_val,no_label_noise=l_noise_val)
    train_loader = DataLoader(train_dat, batch_size=8, shuffle=True, num_workers=8)

    val_dat = ValDataset(transform = composed_notf)
    val_loader = DataLoader(val_dat, batch_size=8, shuffle=True, num_workers=8)

    return train_dat,train_loader, val_loader
                
    
def validation(epoch, model, test_loader, criterion ,args):

    device ='cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    correct = 0
    total = 0

    #loss_val = AverageMeter()
    with torch.no_grad():
        for batch_idx, i in enumerate(test_loader):
            
            inputs = i['x_i']
            targets = i['label']
            # targets = targets.repeat(2).long()
            inputs, targets = inputs.to(device), targets.to(device).long()
            # with torch.no_grad()
            _,output = model(inputs)

            loss = criterion(output, targets)
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.0 * correct / total
    print("Validation accuracy: ",acc)
    
    if acc > args.best_acc:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "acc": acc,
            "epoch": epoch,}

        torch.save(state, os.path.join(folder_name,"memory_bank_supconmodel.pth"))
        args.best_acc = acc
    
    return loss, acc


def main():

    device ='cuda' if torch.cuda.is_available() else 'cpu'
    opt = parse_option()
    opt.best_acc=0.0
    model = set_model(opt)
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    print("Warmup Phase")
    optimizer = set_optimizer(opt,model) 
    train_data,train_loader, val_loader = set_loader(opt)
    print(len(train_loader))

    model.train()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file_name = os.path.join(folder_name,'details.csv')
    print("Experiment settings - Epochs {} , Optimizer {} ,OOD {}, Label noise {}".format(opt.epochs,optimizer,ood_noise_val,l_noise_val), file = open(output_file_name,'w'))

    ###########################################################################################################################################
    
    print("In Training")

    for epoch in range(1, opt.epochs+1):
        
        time1 = time.time()
        adjust_learning_rate(opt, optimizer, epoch)
        end = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_val = AverageMeter()
        loss_main = AverageMeter()
        cross_entropy_loss = nn.CrossEntropyLoss()
        model.train()
        correct = 0
        total = 0
        start = 0
        for idx, sample in enumerate(train_loader,0):

            x_i,x_j,labels,noisy_idx,path_i = sample['x_i'],sample['x_j'],sample['label'],sample['type'],sample['path']
            images = x_i
            images = images.to(device)
            labels = labels.to(device).long()
            bsz = labels.shape[0]
            _,out = model(images)
            
            loss_weighted = cross_entropy_loss(out,labels).to(device)
            loss_main.update(loss_weighted.item())

            optimizer.zero_grad()
            _, predicted = out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
            loss_weighted.backward()
            optimizer.step()

        train_acc = 100.0 * correct / total
        print('Train: {0}\t' 'Weighted Cross entropy loss = {loss.val:.3f}, train accuracy = {1}\t, loss average =({loss.avg:.3f})\t'.format(epoch, train_acc, loss=loss_main))
        sys.stdout.flush()

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        # save the last model
        save_file = os.path.join(
            opt.save_folder, 'last.pth')
        save_model(model, optimizer, opt, opt.epochs, save_file)

        loss_v, val_acc = validation(epoch,model, val_loader, cross_entropy_loss, opt)

        # tensorboard logger
        logger.log_value('train_loss', loss_weighted, epoch)
        logger.log_value('train_accuracy', train_acc, epoch)    
        logger.log_value('val_loss', loss_v, epoch)
        logger.log_value('val_accuracy', val_acc, epoch)

if __name__=="__main__":
    main()

