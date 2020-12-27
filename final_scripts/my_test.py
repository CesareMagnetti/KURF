
"""
infer on ITK images and plot them
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
from fetalnav.datasets.itk_metadata_classification import ITKMetaDataClassification
from fetalnav.models.spn_models import *
sys.path.append("/home/cm19/Code/SPN.pytorch/spnlib")
from spn import *
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



#define train and validate transforms
train_transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale, flip])
validate_transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale])



#SETUP TRAIN AND VALIDATE LOADERS:
#directory path to images
data_dir = "/home/cm19/Documents/Data/webcam_nico/fetalnav_miccai2018/data/iFIND2-MICCAI18/cartesian/all"
#data_dir = "/home/cm19/Documents/Data/placenta"
#load datasets
test_ds = ITKMetaDataClassification(root = data_dir, mode = "infer", transform = train_transform)


# create data loaders
test_loader = torch.utils.data.DataLoader(test_ds,
                                           batch_size=1, num_workers=4, shuffle = True)

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

#load model state_dict (we only need to test, to resume training from checkpoint, load optmizer_state_dict and loss)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

#functions to detach sp map from output
def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap


def generate_outputs(model, input):

    if input.ndimension() == 3:
        input = input.unsqueeze(0)
    assert input.size(0) == 1, 'Batch processing is currently not supported'

    # enable spn inference mode
    model = hook_spn(model)
    # predict scores
    scores = torch.nn.Softmax(dim=1)(model(input)).data.cpu().squeeze()
    # instantiate maps
    maps = torch.zeros((len(scores), input.size(2), input.size(3) ))
    # localize objects
    for class_idx in range(len(scores)):
        # extract class response
        m = model.class_response_maps[0, class_idx].unsqueeze(0).unsqueeze(0)
        # upsample response to input size
        m = F.upsample(m, size=(input.size(2), input.size(3)), mode='bilinear')
        # index response to array
        maps[class_idx] = m.data.cpu().squeeze()

    return scores, maps

#TEST
#set model in evaluation mode
model.eval()
for idx in range(100):
    with torch.no_grad():

        fig, axes = plt.subplots(1, len(classes))
        fig.set_size_inches([3 * len(classes), 3])
        inputs = next(iter(test_loader))
        vin = Variable(inputs.type(torch.FloatTensor))
        in_image = vin.data.numpy().squeeze()
        sc, maps = generate_outputs(model, vin)
        for cl in range(len(classes)):
            response = maps[cl]
            ax = axes[cl]
            ax.imshow(in_image, cmap='gray')
            alpha = .01
            if sc[cl] > 0.01:
                alpha = .7
            ax.imshow(response, cmap=transparent_cmap(plt.cm.jet), alpha=alpha)
            ax.axis('off')
            ax.set_title('{}: {:.1f}'.format(classes[cl], sc[cl]))
        plt.show()
