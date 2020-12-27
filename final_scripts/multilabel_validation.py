"""
this script aim was to try to get a multiple label validation on images trained on single labels,
it would have been intersting to understand if a single label trained model could have had good multilabel results,
but due to time issues i dropped this project. this script uses the multilabel version of the ITKImageclassifier,
stored in file "itk_metadata_classification_cesare.py"

Author: Cesare Magnetti <cesare.magnetti98@gmail.com>
King's College London, UK

"""

#utilities
import numpy as np
from matplotlib import pyplot as plt
import sys
#torch
import torch
from torch.backends import cudnn
from torchvision import transforms as torchtransforms
from torch.nn import functional as F
from torch.autograd import Variable

#fetalnav:
from fetalnav.transforms import itk_transforms as itktransforms
from fetalnav.transforms import tensor_transforms as tensortransforms
from itk_metadata_classification_cesare import ITKMetaDataClassification
from fetalnav.models.spn_models import *
sys.path.append("/home/cm19/Code/SPN.pytorch/spnlib")
from spn import *
########################################################################################################################

#DATA DIRECTORIES
#directory path to images
data_dir = "/home/cm19/Documents/Data/webcam_nico/fetalnav_miccai2018/data/iFIND2-MICCAI18/cartesian/small"


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



#define transform
transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale])

test_ds = ITKMetaDataClassification(root = data_dir, mode = "validate", transform = transform)

# create data loaders
test_loader = torch.utils.data.DataLoader(test_ds,
                                           batch_size=1, num_workers=0, shuffle = True)

# set model class to load it from checkpoint
classes = ['Abdomen', 'Background','Head', 'Limbs','Placenta', 'Spine', 'Thorax']
num_classes = len(classes)
model = resnet18_sp(num_classes, num_maps=512, in_channels=1)


if torch.cuda.is_available():

    test_loader.pin_memory = True
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model).cuda()

#path of model to load
PATH = "/home/cm19/Code/models/resnet18/resnet18_checkpoint_30.pth.tar"

#load model state_dict
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

#TEST

# set up variables
nsamples = len(test_loader.dataset)
nclasses = len(test_loader.dataset.get_classes())
class_correct = [0.] * nclasses
class_total = [0.] * nclasses
multi_class_correct = [0.] * nclasses
multi_class_total = [0.] * nclasses


#set model in evaluation mode
model.eval()
for batch_idx, (data,label,labels) in enumerate(test_loader):

    data, label, labels = data.to(device), label.to(device), labels.to(device)
    data = data.type(torch.FloatTensor)
    with torch.no_grad():
        output = model(data)

    output1 = rescale(output)
    output2 = torch.nn.Softmax(dim=1)(model(data))

    #get number of labels in image
    nlabels = 0
    for n in range(nclasses):
        if labels.cpu().numpy().squeeze()[n] == 1.:
            nlabels += 1

    #single-label accuracy:
    # get accuracy
    _, gt = torch.max(label, 1)
    _, predicted = torch.max(output, 1)

    c = [float(predicted[i] == gt[i]) for i in range(len(gt))]
    for i in range(len(c)):
        index = int(gt[i])
        class_correct[index] += c[i]
        class_total[index] += 1

    #multi label accuracy:
    _, gt = torch.topk(labels, nlabels,largest = True)
    _, predicted = torch.topk(output, nlabels,largest = True)
    gt, predicted = torch.sort(gt), torch.sort(predicted)


    print(gt[0])
    c = [float(gt[0].cpu().numpy().squeeze()[i] == predicted[0].cpu().numpy().squeeze()[i]) for i in range(nlabels)]
    for i in range(len(c)):
        index = int(gt[0].cpu().numpy().squeeze()[i]-)
        multi_class_correct[index] += c[i]
        multi_class_total[index] += 1



for i in range(nclasses):
    print('SINGLE LABEL ACCURACY:\n\nAccuracy of {} : {}%'.format(
            test_loader.dataset.get_classes()[i], 100*class_correct[i]/class_total[i]))
    print('MULTI LABEL ACCURACY:\n\nAccuracy of {} : {}%'.format(
                test_loader.dataset.get_classes()[i], 100*multi_class_correct[i]/multi_class_total[i]))
