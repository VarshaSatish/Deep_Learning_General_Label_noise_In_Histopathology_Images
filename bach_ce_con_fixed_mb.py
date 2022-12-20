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
from custom_loss import SupConLoss,SupervisedContrastiveLoss
from torch.utils.data import Dataset, DataLoader
from dataloader_bach import TrainDataset, ValDataset
from sklearn.neighbors import LocalOutlierFactor
from numpy.linalg import norm
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses,miners,reducers
import pandas as pd
import myTransforms
random_seed = 1 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EXPT_NO = 30
TRIAL = 1
OPTIMIZER = "sgd"
miner_num = 6
learning_rate = 0.01
folder_name ="./Expt_{}_{}_trial_{}_miner_batch_{}".format(EXPT_NO,OPTIMIZER,TRIAL,miner_num)

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
    
    parser.add_argument('--warmup_epochs', type=int, default=10,
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
        format(EXPT_NO, opt.dataset, opt.model, opt.learning_rate, OPTIMIZER, TRIAL)
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

    pre_model = CustomModel()
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        pre_model = apex.parallel.convert_syncbn_model(pre_model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pre_model = CustomModel()
    trained_model = torch.load("/home/varsha/Supervised_SimCLR/varsha/SupContrast/tenpercent_resnet18.ckpt", map_location=torch.device('cpu'))

    
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
    criterion = criterion.to(device)
    cudnn.benchmark = True

    return model, criterion

def set_loader(opt):
    # construct data loader

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    composed_tf = transforms.Compose([myTransforms.RandomHorizontalFlip(),
                                        myTransforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1),
                                        myTransforms.RandomVerticalFlip(),
                                        myTransforms.Resize((224,224)),
                                        myTransforms.ToTensor(),
                                        #myTransforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                                        #RandomElastic(alpha=2, sigma=0.06),
                                        myTransforms.Normalize(mean = mean, std = std)
                                      ])
    composed_notf = transforms.Compose([myTransforms.Resize((224,224)),
                                        myTransforms.ToTensor(),
                                        myTransforms.Normalize(mean = mean, std = std)
                                      ])

    train_dat = TrainDataset(transform = composed_tf,no_ood_noise=15,no_label_noise=14)
    train_loader = DataLoader(train_dat, batch_size=8, shuffle=True, num_workers=1)

    val_dat = ValDataset(transform = composed_notf)
    val_loader = DataLoader(val_dat, batch_size=8, shuffle=True, num_workers=1)

    return train_dat,train_loader, val_loader

def create_memory_bank(model,device):
    
    features={}
    count={}

    with open('BACH_train_file.txt','r') as f:
        x =f.readlines()

    data={}
    j=0

    classes = {'Benign':0.0,'Invasive':1.0,'Normal':2.0,'InSitu':3.0}

    for i in x:
        index_num = i.split(',')[1].lstrip().strip("'")
        types = i.split(',')[2][2:4]
        index =classes[index_num]
        if(types=='id'):
            if(index not in data):
                data[index]=[j]
            else:
                data[index].append(j)
        j+=1
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    composed_tf = transforms.Compose([myTransforms.RandomHorizontalFlip(),
                                        myTransforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1),
                                        myTransforms.RandomVerticalFlip(),
                                        myTransforms.Resize((224,224)),
                                        myTransforms.ToTensor(),
                                        #myTransforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                                        #RandomElastic(alpha=2, sigma=0.06),
                                        myTransforms.Normalize(mean = mean, std = std)
                                      ])
    
    train_data = TrainDataset(transform=composed_tf,no_ood_noise=15,no_label_noise=14)
       
    num_data_points=5
    for i in (0,1,2):
        trainset1 = torch.utils.data.Subset(train_data , data[i][0:num_data_points])
        train_loader = DataLoader(trainset1,batch_size=1,shuffle=False,num_workers=1)
        for idx, sample in enumerate(train_loader,0):
            x_i,x_j,labels,noisy_idx = sample['x_i'],sample['x_j'],sample['label'],sample['type']
            images = x_i
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]
            feature,out = model(images)
            feature = feature.squeeze(0)
            if(idx==0):
                features[i] = [feature]
            else:
                features[i].append(feature)
    return features
                
def update_memory_bank(memory_bank,data,labels,mask_index):
    mask_index = mask_index.tolist() 
    for i in range(data.shape[0]):
        if i not in mask_index:
            #print(data[10])
            index = labels[i].item()
            data[i] = data[i].unsqueeze(0)
            memory_bank[index].append(data[i])
            if (len(memory_bank[index])>300):
                memory_bank[index] = memory_bank[index][-300:]
    
    return memory_bank
        
def calc_similarity(feat,feature,labels,noisy_idx):
    sigma =1.0
    cos =nn.CosineSimilarity(dim=1,eps=1e-6)
    sim1= []

    for i in range(0,feature.shape[0]):
        sum_sim = 0
        index = feat[int(labels[i].item())]
        mean_vec = torch.mean(torch.stack(index), dim=0)
        sim1.append(cos(mean_vec.unsqueeze(0), feature[i].unsqueeze(0)).item())
    
    return sim1
    
def validation(epoch, model, test_loader, criterion ,args):

    device ='cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    correct = 0
    total = 0

    #loss_val = AverageMeter()
    with torch.no_grad():
        for batch_idx, i in enumerate(test_loader):
            
            inputs = torch.cat((i['x_i'],i['x_j']))
            targets = i['label']
            targets = targets.repeat(2).long()
            inputs, targets = inputs.to(device), targets.to(device)
            # with torch.no_grad()
            _,output = model(inputs)

            loss = criterion(output, targets)
            #loss_val.update(loss.item())
            #test_loss += loss.item()
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

        # if not os.path.isdir("checkpoint"):
        #     os.mkdir("checkpoint")
        torch.save(state, os.path.join(folder_name,"memory_bank_supconmodel.pth"))
        args.best_acc = acc
    
    return loss, acc

def print_similarity(train_loader,model,device,memory_bank,iteration,output_file_name):

    sim_vals = {1:[],2:[],3:[]}
    print("Epoch: ",iteration,file=open(output_file_name,"a"))
    for idx, sample in enumerate(train_loader, 0):

        x_i,x_j,labels,noisy_idx = sample['x_i'],sample['x_j'],sample['label'],sample['type']
        images = torch.cat([x_i, x_j], dim=0)
        images = images.to(device)
        labels = labels.to(device)
        bsz = labels.shape[0]
        
        embeddings,out= model(images)
        label = labels.repeat(2)
        noisy_idxs=noisy_idx.repeat(2)
        
        f1, f2 = torch.split(embeddings, [bsz, bsz], dim=0)
        sim1 = calc_similarity(memory_bank,embeddings,label,noisy_idxs)

        for i in range(0,noisy_idxs.shape[0]):
            sim_vals[noisy_idxs[i].item()].append(sim1[i])
            #print("Label: ",label[i].item()," Similarity: ",sim1[i]," Noisy_idx: ",noisy_idxs[i].item() ,file=open(output_file_name,"a"))
  
    print("Noisy index mean similarity score",iteration,file=open(output_file_name,"a"))
    print("\n",file=open(output_file_name,"a"))
    for keys,values in sim_vals.items():
        print("Noisy_idx: ",keys," Mean Similarity Score: ",sum(values)/len(values),file=open(output_file_name,"a"))

def main():

    device ='cuda' if torch.cuda.is_available() else 'cpu'
    opt = parse_option()
    opt.best_acc=0.0
    model,criterion = set_model(opt)
    memory_bank = create_memory_bank(model,device)
    
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    print("Warmup Phase")
    optimizer = set_optimizer(opt,model) 
    train_data,train_loader, val_loader = set_loader(opt)
    print(len(train_loader))

    cross_entropy_loss = nn.CrossEntropyLoss()
    thresh=1.5
    orig_thresh = thresh
    lam = 0.8
    threshold=0.8
    reducer = reducers.ThresholdReducer(low=0, high=threshold)
    loss_func = losses.ContrastiveLoss(distance = CosineSimilarity(),pos_margin=1,neg_margin=0,reducer=reducer)
    miner = miners.MaximumLossMiner(loss_func,output_batch_size=miner_num)
    model.train()

    flag=0
    prev_flag=0
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file_name = os.path.join(folder_name,'similarity_scores.csv'+timestr)
    
    print("Current similarity scores")

    print_similarity(train_loader,model,device,memory_bank,0,output_file_name)
       
    for epoch in range(1, opt.warmup_epochs+1):
        LOF = AverageMeter()
        adjust_learning_rate(opt, optimizer, epoch)
        lof = []
        # train for one epoch
        time1 = time.time()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_val = AverageMeter()
        
        label = torch.Tensor()
        feat = torch.Tensor()
        noisy_idx=torch.Tensor()
        #print("epoch:")
        end = time.time()
        flag=0

        for idx, sample in enumerate(train_loader, 0):
            x_i,x_j,labels,noisy_idx = sample['x_i'],sample['x_j'],sample['label'],sample['type']
            images = torch.cat([x_i, x_j], dim=0)
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]
            embeddings,out= model(images)
            label = labels.repeat(2).long()
            noisy_idxs=noisy_idx.repeat(2)
            hard_pairs = miner(embeddings,label)

            label=label.to(device)
            loss = cross_entropy_loss(out,label)
            loss_val.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            memory_bank =  update_memory_bank(memory_bank,embeddings,label,hard_pairs)

    
            if (idx%10==0):
                print('Train: [{0}][{1}/{2}]\t' 'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(epoch, idx + 1, len(train_loader), loss=loss_val))
                sys.stdout.flush()
                labels=labels.to(device)
                noisy_idxs= noisy_idxs.to(device)
                #calc_similarity(memory_bank,embeddings,labels,noisy_idxs)


        print('Train: [{0}][{1}/{2}]\t' 'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(epoch, idx + 1, len(train_loader), loss=loss_val))
        sys.stdout.flush()

        if(epoch%5==0):
            print("Storing the similarity score after epoch:",epoch)
            print_similarity(train_loader,model,device,memory_bank,epoch,output_file_name)


    print("Warmup phase complete")
    print("Saving warmup phase model")

    torch.save({'epoch': opt.warmup_epochs,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': loss,
                 'memory_bank' : memory_bank
                 }, os.path.join(folder_name, 'warmup_phase_model.pth'))

    print("Warmup Phase Done")
    
    ###########################################################################################################################################
    
    print("Storing Final Similarity scores")

    checkpoint = torch.load(os.path.join(folder_name, 'warmup_phase_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    memory_bank = checkpoint['memory_bank']
    model= model.to(device)

    for idx, sample in enumerate(train_loader,0):

        x_i,x_j,labels,noisy_idx,path_idx = sample['x_i'],sample['x_j'],sample['label'],sample['type'],sample['path']
        images = torch.cat([x_i, x_j], dim=0)
        images = images.to(device)
        labels = labels.to(device)
        bsz = labels.shape[0]

        feature,out = model(images)
        label = labels.repeat(2)
        noisy_idxs = noisy_idx.repeat(2)
        sim1 = calc_similarity(memory_bank,feature,label,noisy_idxs)

        i_idx = torch.Tensor()
       
        for num in range(bsz):
            
            i = labels[num].item()
            i_path = path_idx[num]
            weights_i = (sim1[num] + sim1[num+bsz])/2
            with open(os.path.join(folder_name,"weight_scores.csv"),"a") as f:
                f.write('{} ,{}'.format(weights_i, i_path))
                f.write("\n")

    df = pd.read_csv(os.path.join(folder_name,"weight_scores.csv"),header=None)
    df.to_csv(os.path.join(folder_name,"weight_scores.csv"), header=["score","path"],index=False)

    ###########################################################################################################################################

    print("In Training")

    train_data.change_csv(filecsv = os.path.join(folder_name,"weight_scores.csv"))
    
    print_similarity(train_loader,model,device,memory_bank,opt.epochs,output_file_name)

    for epoch in range(1, opt.epochs+1):
        
        time1 = time.time()
        adjust_learning_rate(opt, optimizer, epoch)
        end = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_val = AverageMeter()
        loss_main = AverageMeter()
        model.train()
        cross_entropy_loss = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        start = 0
        for idx, sample in enumerate(train_loader,0):

            x_i,x_j,labels,noisy_idx,path_i,weight = sample['x_i'],sample['x_j'],sample['label'],sample['type'],sample['path'], sample['score']
            images = torch.cat([x_i, x_j], dim=0)
            #images = x_i
            images = images.to(device)
            # print(images.shape)
            labels = labels.to(device)
            bsz = labels.shape[0]
            #sigmoid = nn.Sigmoid()
            weight = [(abs(float(val))) for val in weight]
            weight = torch.Tensor(weight)
            weight = weight.float()
            weights = torch.FloatTensor(weight)
            # print(weights.shape)
            #weights = sigmoid(weights)
            weights = torch.cat([weight, weight], dim=0).float()
            # print(weights.shape)

            _,out = model(images)
            # f1, f2 = torch.split(feature, [bsz, bsz], dim=0)
            # feature = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # feature = feature.squeeze(0).detach().cpu()

            weights = weights.to(device)
            
            labels = torch.cat([labels, labels], dim=0).long()
            criterion_weighted = nn.CrossEntropyLoss(reduction='none')
            # print(out.shape)
            # print(labels.shape)
            loss_weighted = criterion_weighted(out,labels)

            # print(out.shape)
            # print(labels.shape)
            loss_weighted = torch.dot(weights, loss_weighted)
            loss_weighted = loss_weighted/weights.sum()

            loss_main.update(loss_weighted.item())

            optimizer.zero_grad()

            _, predicted = out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
            loss_weighted.backward()
            optimizer.step()

        train_acc = 100.0 * correct / total
        print('Train: {0}\t' 'Weighted Cross entropy loss = {loss.val:.3f}, train accuracy = {1}\t, loss average = ({loss.avg:.3f})\t'.format(epoch, train_acc, loss=loss_main))
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

