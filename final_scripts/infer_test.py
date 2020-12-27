
"""
function to infer on fetal images given a numpy array (image) and the model needed to infer on itAuthor: Cesare Magnetti <cesare.magnetti98@gmail.com>
King's College London, UK

USAGE:
call from terminal like so:
python infer_test.py -i [image] -m [model_name] -mp [model_path]

important info: -you may want to change line 46/47 to fit to your input images


"""

import argparse
import numpy as np
from PIL import Image
import os
import infer
from matplotlib import pyplot as plt
import torch
from torchvision import transforms as torchtransforms
from fetalnav.transforms import tensor_transforms as tensortransforms

parser = argparse.ArgumentParser(description='Webcam Infer')
parser.add_argument('--image', '-i', default='default_image', help='numpy image to infer (must be 1x224x224)')
parser.add_argument('--model-name', '-m', default='resnet18', help='which model to use' )
parser.add_argument('--model-path','-mp', default='default_model_path', help='path to the model file')

args = parser.parse_args()

global path_to_model, image, num_classes

#TRANSFORMS
totensor = torchtransforms.ToTensor()
crop     = tensortransforms.CropToRatio(outputaspect=1.)
resize   = tensortransforms.Resize(size=[224, 224], interp='bilinear')
rescale  = tensortransforms.Rescale(interval=(0, 1))
transform = torchtransforms.Compose([totensor, crop, resize, rescale])

#check input image
if args.image == 'default_image':
    image = np.ones(shape = (224,224))
else:
    #change to fit your input image tipe
    image = rescale(resize(crop(totensor(Image.open(args.image).convert('L')))))
    image = image.numpy().squeeze()
#check model path
if args.model_path == 'default_model_path':
    path_to_model = "/home/cm19/Code/models/resnet18_aug/resnet18_checkpoint_80.pth.tar"
else:
    assert os.path.exists(args.model_path), "ERROR: model path does not exist"
    path_to_model = args.model_path


classes = ['Abdomen', 'Background', 'Head', 'Limbs', 'Placenta', 'Spine', 'Thorax']
num_classes = len(classes)

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

def main():

    model = infer.initialize(path_to_model, args.model_name, num_classes)
    (scores, maps) = infer.do_infer_with_maps(image, model)
    sc = infer.do_infer_without_maps(image, model)

    print('inferred class: {}'.format(classes[torch.argmax(sc)]))
    print('inferred class: {}'.format(classes[torch.argmax(scores)]))

    fig, axes = plt.subplots(1, len(classes))
    fig.set_size_inches([3 * len(classes), 3])

    for cl in range(len(classes)):
        response = maps[cl]
        ax = axes[cl]
        ax.imshow(image, cmap='gray')
        alpha = .01
        if sc[cl] > 0.01:
            alpha = .7
        ax.imshow(response, cmap=transparent_cmap(plt.cm.jet), alpha=alpha)
        ax.axis('off')
        ax.set_title('{}: {:.1f}'.format(classes[cl], sc[cl]))
    plt.show()



if __name__ == '__main__':
    main()