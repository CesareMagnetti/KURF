
"""
train fetalnav model on ITK images
Author: Cesare Magnetti <cesare.magnetti98@gmail.com>
King's College London, UK

"""

#numpy
import numpy as np
#torch
import torch
from torch.backends import cudnn
from torchvision import transforms as torchtransforms

#fetalnav:
from fetalnav.transforms import itk_transforms as itktransforms
from fetalnav.transforms import tensor_transforms as tensortransforms
from fetalnav.datasets.itk_metadata_classification import ITKMetaDataClassification
#from itk_metadata_classification_cesare import ITKMetaDataClassification
from fetalnav.models.spn_models import *
from tqdm import tqdm
import os
import time
import math

#pyplot
from matplotlib import pyplot as plt

#define device type for data & labels
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#SET UP TRANSFORMATIONS:
# create transformation and data augmentation schemes
resample = itktransforms.Resample(new_spacing=[.5, .5, 1.])
tonumpy  = itktransforms.ToNumpy(outputtype='float')
totensor = torchtransforms.ToTensor()
crop     = tensortransforms.CropToRatio(outputaspect=1.)
resize   = tensortransforms.Resize(size=[224, 224], interp='bilinear')
rescale  = tensortransforms.Rescale(interval=(0, 1))
flip     = tensortransforms.Flip(axis=2)

#define train and validate transforms
train_transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale, flip])
validate_transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale])

#SETUP TRAIN AND VALIDATE LOADERS:

#directory paths

data_dir = "/home/cm19/Documents/Data/webcam_nico/fetalnav_miccai2018/data/iFIND2-MICCAI18/cartesian/all"

modelpath = "/home/cm19/Code/models/resnet18/"
modelname = "resnet18_checkpoint_30.pth.tar"
START_FROM_CHECKPOINT = True


#load datasets
train_ds = ITKMetaDataClassification(root = data_dir, mode = "train", transform = train_transform)
validate_ds = ITKMetaDataClassification(root = data_dir, mode = "validate", transform = validate_transform)

#define class cardinality: balance the training dataset
train_cardinality = train_ds.get_class_cardinality()
val_cardinality = validate_ds.get_class_cardinality()
#train_sample_weights = torch.from_numpy(train_ds.get_sample_weights())

print(train_cardinality, train_ds.get_classes())
print('')
print('train-dataset: \n')
for idx, c in enumerate(train_ds.get_classes()):
    print('{}: \t{}'.format(train_cardinality[idx], c))
print('')
print('validate-dataset: \n')
for idx, c in enumerate(validate_ds.get_classes()):
    print('{}: \t{}'.format(val_cardinality[idx], c))
print('')

weights = np.array([0.]*len(train_cardinality))


weights = [sum(train_cardinality)/train_cardinality[idx] for idx in range(len(train_cardinality))]
weights = weights/sum(weights)

print("WEIGHT\t\tCLASS")
for idx, c in enumerate(train_ds.get_classes()):
    print('{:.4f}: \t{}'.format(weights[idx], c))


weights = torch.from_numpy(weights).type(torch.float32)

# create data loaders
train_loader = torch.utils.data.DataLoader(train_ds,
                                           batch_size=128, num_workers=0, shuffle = True)

val_loader = torch.utils.data.DataLoader(validate_ds,
                                         batch_size=64, shuffle=False, num_workers=0)


# class labels
classes = train_ds.get_classes()
num_classes = len(classes)

model = resnet18_sp(num_classes=num_classes, num_maps=512, in_channels=1)
#model = vgg13_sp(num_classes=num_classes, num_maps=512, in_channels=1)
criterion = nn.CrossEntropyLoss(weight=weights)

if torch.cuda.is_available():
    weights = weights.to(device)
    train_loader.pin_memory = True
    val_loader.pin_memory = True
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda()

#optimizer and criterion
optimizer = torch.optim.Adam(model.parameters())

#validate function
def validate(model, loader, device):

    print("validating...", end = '')
    #set model in evaluate mode
    model.eval()

    #set up variables
    nsamples = len(loader.dataset)
    nclasses = len(loader.dataset.get_classes())
    class_correct = [0.] * nclasses
    class_total = [0.] * nclasses

    #forward pass
    for batch_idx, (data, labels) in enumerate(loader):
        # move data and labels to device available
        data, labels = data.to(device), labels.to(device)
        data = data.type(torch.float32)
        labels = labels.type(torch.float32)

        # compute output and loss
        output = model(data)

        #get accuracy
        _, gt = torch.max(labels, 1)
        _, predicted = torch.max(output, 1)

        c = [float(predicted[i] == gt[i]) for i in range(len(gt))]
        for i in range(len(c)):
            index = int(gt[i])
            class_correct[index] += c[i]
            class_total[index] += 1

        # print some sort of loading bar to monitor process
        if batch_idx % 150 == 0:
            print('.', sep='', end='', flush=True)


    for i in range(nclasses):
        print('\nAccuracy of {} : {}%'.format(
                loader.dataset.get_classes()[i], 100*class_correct[i]/class_total[i]))


def save_model(model, epoch, optimizer, loss, name = None, path = None):

    #save model if validation was above a certain threshold
    if name == None:
        name = "resnet18_checkpoint_" + str(epoch+1) + ".pth.tar"
    if path == None:
        path = "/home/cm19/Code/models/vgg/"

    print("saving model state: ", name, "\tto: ", path)

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path + name)


START_EPOCH = 0
MAX_EPOCH = 100
 #load model
if START_FROM_CHECKPOINT:
    path_to_model = modelpath + modelname
    if os.path.exists(path_to_model):
        checkpoint = torch.load(path_to_model)
        if torch.cuda.is_available():
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        START_EPOCH = checkpoint['epoch'] + 1

for epoch in range(START_EPOCH, MAX_EPOCH):

    # set model in train mode
    model.train()

    #train loop (single epoch)
    av_loss = 0
    start_time = time.time()
    for batch_idx, (data, labels) in enumerate(train_loader):
        #move data and labels to device available and cast them to float32
        data, labels = data.to(device), labels.to(device)
        data = torch.autograd.Variable(data).type(torch.float32)
        labels = labels.type(torch.float32)

        data.requires_grad_(True)

        #compute output and loss
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            av_loss+=torch.mean(loss)

        #print some training info:
        if batch_idx % 10 == 0:
            #get current time
            end_time = time.time()
            #elapsed time is current time minus start time
            elapsed_time = end_time - start_time
            #predict time to end of epoch based on the elapsed time measured
            predicted_end = elapsed_time * (len(train_loader) - batch_idx)/10

            min = math.floor(predicted_end/60)
            sec = predicted_end - min*60

            print("\ntrain epoch : {}\tbatch: {}/{}  ({:.0f}%)\tAverage Loss: {:.6f}\tTime: {:.2f} s  --  {:.0f} min {:.2f} s".format(
                    epoch + 1, batch_idx, len(train_loader), 100. * batch_idx/len(train_loader),
                    av_loss.data.item(), elapsed_time, min, sec), end = "\n")

            #set the start time as the current measured time for next checkpoint
            start_time = time.time()

        #print some sort of loading bar to monitor process
        print('-', sep='', end='', flush=True)

    print("\nfinal average loss: {:.6f}".format(av_loss.data.item()))

    # validate every five epochs
    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            validate(model, val_loader, device)
        save_model(model, epoch, optimizer, av_loss, path = modelpath)




