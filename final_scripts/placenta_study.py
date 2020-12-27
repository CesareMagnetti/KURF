
"""
script to track the history of inferred labels for placenta validation images throughout the epochs of a model
placenta images often get confused by the network, this script tries to investigate that confusion

Author: Cesare Magnetti <cesare.magnetti98@gmail.com>
King's College London, UK
"""

#utilities
import os
#torch
import torch
from torch.backends import cudnn
from torchvision import transforms as torchtransforms
#fetalnav:
from fetalnav.transforms import itk_transforms as itktransforms
from fetalnav.transforms import tensor_transforms as tensortransforms
from fetalnav.datasets.itk_metadata_classification import ITKMetaDataClassification
from fetalnav.models.spn_models import *
########################################################################################################################


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


#placenta dataset
validate_transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale])
data_dir = "/home/cm19/Documents/Data/placenta"
placenta_ds = ITKMetaDataClassification(root = data_dir, mode = "validate", transform = validate_transform)


# set model class to load it from checkpoint
classes = ['Abdomen', 'Background','Head', 'Limbs','Placenta', 'Spine', 'Thorax']
num_classes = len(classes)
model = resnet18_sp(num_classes=num_classes, num_maps=512, in_channels=1)
#model = vgg13_sp(num_classes=num_classes, num_maps=512, in_channels=1)

#set cuda if possible
if torch.cuda.is_available():
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model).cuda()

#path of model to load
PATH = "/home/cm19/Code/models/resnet18_aug/"

HISTORY = {}
#initialize dictionary with empty lists
for idx in range(len(placenta_ds)):
    HISTORY["{}".format(placenta_ds.get_filenames()[idx].split('/')[-1])] = []

for epoch in range(1, 91):
    print(epoch)
    #load model state_dict (we only need to test, to resume training from checkpoint, load optmizer_state_dict and loss)
    name = "resnet18_checkpoint_" + str(epoch) + ".pth.tar"
    assert os.path.exists(PATH + name), "model not found/ not existing"
    checkpoint = torch.load(PATH + name)
    model.load_state_dict(checkpoint['model_state_dict'])

    #KEEP TRACK OF PLACENTA OUTCOMES (DATASET ONLY CONTAINS PLACENTA GROUND TRUTHS)
    # set model in evaluate mode
    model.eval()
    for idx in range(len(placenta_ds)):
        # move data and labels to device available
        data, _ = placenta_ds[idx]
        data = data.to(device).type(torch.float32)
        data = data.unsqueeze(0)

        # compute output
        output = model(data)
        _, predicted = torch.max(output, 1)
        #update dictionary
        HISTORY["{}".format(placenta_ds.get_filenames()[idx].split('/')[-1])].append(classes[predicted])




#WRITE RESULTS TO A TEXT FILE
assert os.path.exists("/home/cm19/Code/models/accuracies/placenta_study.txt")
file = open("/home/cm19/Code/models/accuracies/placenta_study.txt", "a")
for line in HISTORY:
    file.write("{}\t{}".format(line, HISTORY[line]))
file.write("\n")
file.close()
