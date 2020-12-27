
"""
script to monitor the accuracy of a model throughout all saved checkpoints (for now trained on raw images) on webcam images
Author: Cesare Magnetti <cesare.magnetti98@gmail.com>
King's College London, UK

USAGE:

call from terminal like so:
python evaluate.py --mode [cam/raw] -m [model name] -e [epochs info (min max step)] --isaug [aug/nonaug] -b [batch_size] -n [num_workers]

FOR SPECIFIC EPOCH MODEL EVALUATION PASS THE WANTED EPOCH IN THE 'epochs_info' ARGUMENT

important info: -CHANGE DATA_ROOT, MODE_ROOT AND FILE_PATH TO YOUR CUSTOM DIRECTORIES FROM LINE 55 TO LINE 65
                -model_name must be either resnet18 or vgg13 [for now]
                -mode_root must contain model checkpoints saved like so: "modelname_checkpoint_epoch_pth.tar"
                -YOU MUST SAVE THE PLOTS MANUALLY WHEN THEY POP ON THE PLOT WINDOW!! program does not automatically save them

"""

#imports

#utilities
import sys
import argparse
import os
from matplotlib import pyplot as plt
#torch
import torch
from torch.backends import cudnn
from torchvision import transforms as torchtransforms
#webcamclassifier
from webcamclassifier import WebcamClassifier
#fetalnav:
from fetalnav.transforms import tensor_transforms as tensortransforms
from fetalnav.datasets.itk_metadata_classification import ITKMetaDataClassification
from fetalnav.transforms import itk_transforms as itktransforms
from fetalnav.models.spn_models import *
########################################################################################################################


#ARGUMENT PARSER
parser = argparse.ArgumentParser(description='Webcam Validation Program')
parser.add_argument('--mode',default='cam', help='either cam or raw')
parser.add_argument('--model', '-m', default='resnet18', help='which model to use' )
parser.add_argument('--epochs-info', '-e', default = 'default_epoch_info', nargs='*', type = int, help = 'min_epoch, max_epoch and optionally checkpoint_step')
parser.add_argument('--isaug', help = 'string, either aug or nonaug')
parser.add_argument('--batch-size', '-b',type=int, default=16, help = 'batch size (default = 16)')
parser.add_argument('--num-workers', '-n',type=int, default=8, help = 'batch size (default = 8)')
args = parser.parse_args()

#ARGUMENTS HANDLING
assert args.mode == 'cam' or args.mode == 'raw', "mode must be cam/raw"

#CHANGE HERE TO YOUR CUSTOM DIRECTORIES!!!!
#add to parser if you want (too many parse arguments, especially paths, can get messy from command line):
if args.mode == 'cam':
    cam_file_path = "/home/cm19/Code/models/accuracies/accuracies/small_dataset/webcam/"
    cam_data_root = "/home/cm19/Documents/Data/webcam_random"
    assert os.path.exists(cam_file_path), "file_path for the directory to save the text file does not exists"
    assert os.path.exists(cam_data_root), "ERROR: data root not found"
else:
    raw_file_path = "/home/cm19/Code/models/accuracies/accuracies/all_dataset/RAW/"
    raw_data_root = "/home/cm19/Documents/Data/webcam_nico/fetalnav_miccai2018/data/iFIND2-MICCAI18/cartesian/all"
    assert os.path.exists(raw_file_path), "file_path for the directory to save the text file does not exists"
    assert os.path.exists(raw_data_root), "ERROR: data root not found"

model_root = "/home/cm19/Code/models/resnet18_aug/"

assert os.path.exists(model_root), "ERROR: model root not found"

assert args.model == "resnet18" or args.model == "vgg13", "ERROR: unknown model input, known models: resnet18|vgg13"

if args.isaug == "aug":
    isaug = "_aug_"
elif args.isaug == "nonaug":
    isaug = "_"
else:
    print("ERROR: isaug argument can only be either aug or nonaug")
    sys.exit()

if args.epochs_info == 'default_epoch_info':
    print('ERROR: for multi-epochs validation u must at list provide a min and max epoch as an argument')
    sys.exit()
else:
    epochs = args.epochs_info
    assert 1 <= len(epochs) <= 3, "ERROR: wrong inputs for the epochs_info parameter, must be at least 2 integers, step is optional"
    if len(epochs) == 1:
        epochs.append(epochs[0])
        epochs.append(1)
    elif len(epochs) == 2:
        #append a step of 1 to epochs
        epochs.append(1)
    else:
        assert (epochs[1]+1-epochs[0])%epochs[2] == 0, "ERROR: inputted epoch step is MUST satisfy: (MAX_EPOCH+1-MIN_EPOCH)%STEP == 0"



def main_cam(root, model_name, model_root, epochs, isaug, file_path, batch, workers):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    #TRANSFORMS
    totensor = torchtransforms.ToTensor()
    crop     = tensortransforms.CropToRatio(outputaspect=1.)
    resize   = tensortransforms.Resize(size=[224, 224], interp='bilinear')
    rescale  = tensortransforms.Rescale(interval=(0, 1))
    transform = torchtransforms.Compose([totensor, crop, resize, rescale])

    #DATASET AND DATALOADER
    ds = WebcamClassifier(root, mode = "validate", transform=transform)
    # create data loader
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, shuffle = False)

    #DEFINE MODEL CLASS
    classes = ds.get_classes()
    num_classes = len(classes)

    if model_name == "resnet18":
        model = resnet18_sp(num_classes=num_classes, num_maps=512, in_channels=1)
    elif model_name == "vgg13":
        model = vgg13_sp(num_classes=num_classes, num_maps=512, in_channels=1)

    #set on GPU
    if torch.cuda.is_available():
        loader.pin_memory = True
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model).cuda()

    #initialize variable to contain class accuracies through all epochs (and loss)
    abdomen = []
    background = []
    head = []
    limbs = []
    placenta = []
    spine = []
    thorax = []
    loss = []

    #set the epochs for which you have a model to test
    for epoch in range(epochs[0], epochs[1]+1, epochs[2]):
        # print some tracking
        print("curently validading epoch: ", epoch)
        # load model state_dict (only works if model was saved as shown below)
        name = model_name + "_checkpoint_" + str(epoch) + ".pth.tar"
        assert os.path.exists(model_root + name), "model not found/ not existing"
        checkpoint = torch.load(model_root + name)
        model.load_state_dict(checkpoint['model_state_dict'])
        loss.append(checkpoint['loss'])

        # TEST MODEL ACCURACY ON UNSEEN DATA
        # set model in evaluate mode
        model.eval()
        # set up variables
        nclasses = len(loader.dataset.get_classes())
        class_correct = [0.] * nclasses
        class_total = [0.] * nclasses

        for batch_idx, (data, labels) in enumerate(loader):

            # move data and labels to device available
            data, labels = data.to(device), labels.to(device)
            data = data.type(torch.float32)
            labels = labels.type(torch.float32)

            # compute output and loss
            output = model(data)

            # get accuracy
            _, gt = torch.max(labels, 1)
            _, predicted = torch.max(output, 1)

            c = [float(predicted[i] == gt[i]) for i in range(len(gt))]
            for i in range(len(c)):
                index = int(gt[i])
                class_correct[index] += c[i]
                class_total[index] += 1

        abdomen.append(100 * class_correct[0] / class_total[0])
        background.append(100 * class_correct[1] / class_total[1])
        head.append(100 * class_correct[2] / class_total[2])
        limbs.append(100 * class_correct[3] / class_total[3])
        placenta.append(100 * class_correct[4] / class_total[4])
        spine.append(100 * class_correct[5] / class_total[5])
        thorax.append(100 * class_correct[6] / class_total[6])

        # print results in a text file
        file_name = model_name + isaug + str(epochs[0]) + "_" + str(epochs[1]) + "_cam.txt"
        #assert os.path.exists("/home/cm19/Code/models/accuracies/accuracies/small_dataset/webcam/vgg13_1_44_accuracies_cam.txt")
        file = open(file_path + file_name, "a+")
        file.write("\n\nepoch: " + str(epoch) + "\n\n")
        for i in range(nclasses):
            file.write("Accuracy of {} : {}%\n\n".format(
                loader.dataset.get_classes()[i], 100 * class_correct[i] / class_total[i]))
        file.write("\n\n")
        file.close()

    # plot the accuracies vs the epochs

    fig = plt.figure(1)
    plt.plot(range(epochs[0], epochs[1]+1, epochs[2]), abdomen, c='b', label='abdomen')
    plt.plot(range(epochs[0], epochs[1]+1, epochs[2]), background, c='g', label='background')
    plt.plot(range(epochs[0], epochs[1]+1, epochs[2]), head, c='r', label='head')
    plt.plot(range(epochs[0], epochs[1]+1, epochs[2]), limbs, c='c', label='limbs')
    plt.plot(range(epochs[0], epochs[1]+1, epochs[2]), placenta, c='m', label='placenta')
    plt.plot(range(epochs[0], epochs[1]+1, epochs[2]), spine, c='y', label='spine')
    plt.plot(range(epochs[0], epochs[1]+1, epochs[2]), thorax, c='k', label='thorax')
    plt.legend(loc='upper left')
    plt.show()

    fig1 = plt.figure(2)
    plt.plot(range(epochs[0], epochs[1]+1, epochs[2]), loss, c='b', label='loss')
    plt.legend(loc='upper left')
    plt.show()

def main_raw(root, model_name, model_root, epochs, isaug, file_path, batch, workers):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # SET UP TRANSFORMATIONS:
    resample = itktransforms.Resample(new_spacing=[.5, .5, 1.])
    tonumpy = itktransforms.ToNumpy(outputtype='float')
    totensor = torchtransforms.ToTensor()
    crop = tensortransforms.CropToRatio(outputaspect=1.)
    resize = tensortransforms.Resize(size=[224, 224], interp='bilinear')
    rescale = tensortransforms.Rescale(interval=(0, 1))
    transform = torchtransforms.Compose([resample, tonumpy, totensor, crop, resize, rescale])

    # DATASET AND DATALOADER
    ds = ITKMetaDataClassification(root, mode="validate", transform=transform)
    # create data loader
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, shuffle=False)

    # DEFINE MODEL CLASS
    classes = ds.get_classes()
    num_classes = len(classes)

    if model_name == "resnet18":
        model = resnet18_sp(num_classes=num_classes, num_maps=512, in_channels=1)
    elif model_name == "vgg13":
        model = vgg13_sp(num_classes=num_classes, num_maps=512, in_channels=1)

    # set on GPU
    if torch.cuda.is_available():
        loader.pin_memory = True
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model).cuda()

    # initialize variable to contain class accuracies through all epochs (and loss)
    abdomen = []
    background = []
    head = []
    limbs = []
    placenta = []
    spine = []
    thorax = []
    loss = []

    # set the epochs for which you have a model to test
    for epoch in range(epochs[0], epochs[1] + 1, epochs[2]):
        # print some tracking
        print("curently validading epoch: ", epoch)
        # load model state_dict (only works if model was saved as shown below)
        name = model_name + "_checkpoint_" + str(epoch) + ".pth.tar"
        assert os.path.exists(model_root + name), "model not found/ not existing"
        checkpoint = torch.load(model_root + name)
        model.load_state_dict(checkpoint['model_state_dict'])
        loss.append(checkpoint['loss'])

        # TEST MODEL ACCURACY ON UNSEEN DATA
        # set model in evaluate mode
        model.eval()
        # set up variables
        nclasses = len(loader.dataset.get_classes())
        class_correct = [0.] * nclasses
        class_total = [0.] * nclasses

        for batch_idx, (data, labels) in enumerate(loader):

            # move data and labels to device available
            data, labels = data.to(device), labels.to(device)
            data = data.type(torch.float32)
            labels = labels.type(torch.float32)

            # compute output and loss
            output = model(data)

            # get accuracy
            _, gt = torch.max(labels, 1)
            _, predicted = torch.max(output, 1)

            c = [float(predicted[i] == gt[i]) for i in range(len(gt))]
            for i in range(len(c)):
                index = int(gt[i])
                class_correct[index] += c[i]
                class_total[index] += 1

        abdomen.append(100 * class_correct[0] / class_total[0])
        background.append(100 * class_correct[1] / class_total[1])
        head.append(100 * class_correct[2] / class_total[2])
        limbs.append(100 * class_correct[3] / class_total[3])
        placenta.append(100 * class_correct[4] / class_total[4])
        spine.append(100 * class_correct[5] / class_total[5])
        thorax.append(100 * class_correct[6] / class_total[6])

        # print results in a text file
        file_name = model_name + isaug + str(epochs[0]) + "_" + str(epochs[1]) + "_" + args.mode + "_.txt"
        # assert os.path.exists("/home/cm19/Code/models/accuracies/accuracies/small_dataset/webcam/vgg13_1_44_accuracies_cam.txt")
        file = open(file_path + file_name, "a+")
        file.write("\n\nepoch: " + str(epoch) + "\n\n")
        for i in range(nclasses):
            file.write("Accuracy of {} : {}%\n\n".format(
                loader.dataset.get_classes()[i], 100 * class_correct[i] / class_total[i]))
        file.write("\n\n")
        file.close()

    # plot the accuracies vs the epochs

    fig = plt.figure(1)
    plt.plot(range(epochs[0], epochs[1] + 1, epochs[2]), abdomen, c='b', label='abdomen')
    plt.plot(range(epochs[0], epochs[1] + 1, epochs[2]), background, c='g', label='background')
    plt.plot(range(epochs[0], epochs[1] + 1, epochs[2]), head, c='r', label='head')
    plt.plot(range(epochs[0], epochs[1] + 1, epochs[2]), limbs, c='c', label='limbs')
    plt.plot(range(epochs[0], epochs[1] + 1, epochs[2]), placenta, c='m', label='placenta')
    plt.plot(range(epochs[0], epochs[1] + 1, epochs[2]), spine, c='y', label='spine')
    plt.plot(range(epochs[0], epochs[1] + 1, epochs[2]), thorax, c='k', label='thorax')
    plt.legend(loc='upper left')
    plt.show()

    fig1 = plt.figure(2)
    plt.plot(range(epochs[0], epochs[1] + 1, epochs[2]), loss, c='b', label='loss')
    plt.legend(loc='upper left')
    plt.show()



if __name__ == '__main__':
    if args.mode == 'cam':
        main_cam(cam_data_root, args.model, model_root, epochs, isaug, cam_file_path, args.batch_size, args.num_workers)
    else:
        main_raw(raw_data_root, args.model, model_root, epochs, isaug, raw_file_path, args.batch_size, args.num_workers)
