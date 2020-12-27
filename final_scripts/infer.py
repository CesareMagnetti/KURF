"""
class to infer on numpy images (1 channel batch size = 1), to be called in 'infer_test.py'
Author: Cesare Magnetti <cesare.magnetti98@gmail.com>
King's College London, UK

important info:

"""
import os
import sys
import torch
from torch.nn import functional as F
from fetalnav.transforms import tensor_transforms as tensortransforms
from torchvision import transforms as torchtransforms
from fetalnav.models.spn_models import *
sys.path.append("/home/cm19/Code/SPN.pytorch/spnlib")
from spn import *

#define useful transforms
rescale = tensortransforms.Rescale(interval=(0, 1))
flip = tensortransforms.Flip(axis=2)
transform = torchtransforms.Compose([rescale, flip])

def initialize(path_to_model, model_name, num_classes):

    # check input model
    if model_name == 'resnet18':
        model = resnet18_sp(num_classes, num_maps=512, in_channels=1)
    elif model_name == 'resnet34':
        model = resnet34_sp(num_classes, num_maps=512, in_channels=1)
    elif model_name == 'vgg13':
        model = vgg13_sp(num_classes, batch_norm=False, num_maps=512, in_channels=1)
    elif model_name == 'vgg13_bn':
        model = vgg13_sp(num_classes, batch_norm=True, num_maps=512, in_channels=1)
    elif model_name == 'vgg16':
        model = vgg16_sp(num_classes, batch_norm=False, num_maps=512, in_channels=1)
    elif model_name == 'vgg16_bn':
        model = vgg16_sp(num_classes, batch_norm=True, num_maps=512, in_channels=1)
    elif model_name == 'alexnet':
        model = alexnet_sp(num_classes, num_maps=512, in_channels=1)
    else:
        print(
            'No network known: {}, possible choices are resnet18|resnet34|vgg13|vgg13_bn|vgg16|vgg16_bn|alexnet'.format(model_name))
        sys.exit(0)

    # set model on GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # set model in evaluation mode
    model.eval()

    #load model
    assert os.path.exists(path_to_model), "ERROR: specified path does not exist"
    checkpoint = torch.load(path_to_model)

    if torch.cuda.is_available():
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    return model


#INFER FUNCTION WITH SOFT PROPOSAL MAPS
def do_infer_with_maps(image, model):

    image = torch.from_numpy(image)
    image = transform(image)
    if torch.cuda.is_available():
        image = image.to(torch.device("cuda:0"))
    image = image.unsqueeze(0).unsqueeze(0)

    # elif image.dim() == 3:
    #     # convert rgb to grayscale
    #     image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    #     # convert to tensor and unsqueeze()
    #     image = torch.from_numpy(image)
    #     if torch.cuda.is_available():
    #         image = image.to(torch.device("cuda:0"))
    #     image = image.unsqueeze(0).unsqueeze(0)


    image = torch.autograd.Variable(image).type(torch.float32)
    # enable spn inference mode
    model = hook_spn(model)
    # predict scores
    scores = torch.nn.Softmax(dim=1)(model(image)).data.cpu().squeeze()
    # instantiate maps
    maps = torch.zeros((len(scores), image.size(2), image.size(3)))
    # localize objects
    for class_idx in range(len(scores)):
        # extract class response
        m = model.class_response_maps[0, class_idx].unsqueeze(0).unsqueeze(0)
        # upsample response to input size
        m = F.upsample(m, size=(image.size(2), image.size(3)), mode='bilinear')
        # index response to array
        maps[class_idx] = m.data.cpu().squeeze()

    return scores, maps



#INFER WITHOUT SOFT PROPOSAL MAPS
def do_infer_without_maps(image, model):

    image = torch.from_numpy(image)
    image = transform(image)
    if torch.cuda.is_available():
        image = image.to(torch.device("cuda:0"))
    image = image.unsqueeze(0).unsqueeze(0)

    # elif image.dim() == 3:
    #     #convert rgb to grayscale
    #     image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    #     #convert to tensor and unsqueeze()
    #     image = torch.from_numpy(image)
    #     if torch.cuda.is_available():
    #         image = image.to(torch.device("cuda:0"))
    #     image = image.unsqueeze(0).unsqueeze(0)

    image = torch.autograd.Variable(image).type(torch.float32)
    # predict scores
    return torch.nn.Softmax(dim=1)(model(image)).data.cpu().squeeze()