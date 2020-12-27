"""
class to build a dataset for webcam images
Author: Cesare Magnetti <cesare.magnetti98@gmail.com>
King's College London, UK

"""
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
from torch.utils.data import Dataset
import torch

IMG_EXTENSIONS = ['.png', '.jpg']

def _is_image_file(filename):
    """
    Is the given extension in the filename supported ?
    """
    # FIXME: Need to add all available PIL image types!
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_image(fname):
    """
    Load supported image and return the loaded image.
    """
    return Image.open(fname).convert('L')


def save_image(img, fname):
    """
    Save image with the given filename (full path)
    """
    img.save(fname)

def extractlabelfromfile(fname):
    '''
    infer label from full image path (supposing images are divided in folders which names are the labels)
    '''
    return fname.split('/')[-2]

def find_classes(filenames):
    classes = [extractlabelfromfile(f) for f in filenames]
    classes = list(set(classes))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def calculate_class_cardinality(filenames):
    classes = [extractlabelfromfile(f) for f in filenames]
    _, counts = np.unique(classes, return_counts=True)
    return counts

def calculate_sample_weights(filenames):
    cardinality = calculate_class_cardinality(filenames)
    weights = np.array([0.] * len(cardinality))
    weights = [sum(cardinality) / cardinality[idx] for idx in range(len(cardinality))]
    weights = weights / sum(weights)
    return weights



#CLASSIFIER



class WebcamClassifier(Dataset):
    """
       Arguments
       ---------
       root : string
           Root directory of dataset. The folder should contain all images for each
           mode of the dataset ('train', 'validate', or 'infer'). Each mode-version
           of the dataset should be in a subfolder of the root directory

           The images can be in any PIL readable format (e.g. .png/.jpg)
           For the 'train' and 'validate' modes, it is assume that images are stored in folders named after their labels

       mode : string, (Default: 'train')
           'train', 'validate', or 'infer'
           Loads data from these folders.
           train and validate folders both must contain subfolders images and labels while
           infer folder needs just images subfolder.
       transform : callable, optional
           A function/transform that takes in input itk image or Tensor and returns a
           transformed
           version. E.g, ``transforms.RandomCrop``

       """

    def __init__(self, root, mode='train', transform=None, target_transform=None):
        # training set or test set

        assert (mode in ['train', 'validate', 'infer'])

        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(root, 'train')
        elif mode == 'validate':
            self.root = os.path.join(root, 'validate')
        else:
            self.root = os.path.join(root, 'infer') if os.path.exists(os.path.join(root, 'infer')) else root

        def gglob(path, regexp=None):
            """Recursive glob
            """
            import fnmatch
            import os
            matches = []
            if regexp is None:
                regexp = '*'
            for root, dirnames, filenames in os.walk(path, followlinks=True):
                for filename in fnmatch.filter(filenames, regexp):
                    matches.append(os.path.join(root, filename))
            return matches

        # Get filenames of all the available images
        self.filenames = [y for y in gglob(self.root, '*.*') if _is_image_file(y)]

        if len(self.filenames) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.root + "\n"
                                                                                  "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.filenames.sort()

        self.transform = transform
        self.target_transform = target_transform

        classes, class_to_idx = find_classes(self.filenames)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.class_cardinality = calculate_class_cardinality(self.filenames)
        self.sample_weights = calculate_sample_weights(self.filenames)


    def __getitem__(self, index):
        """
        Arguments
        ---------
        index : int
            index position to return the data
        Returns
        -------
        tuple: (image, label) where label the organ apparent in the image
        """

        image = load_image(self.filenames[index])


        label = extractlabelfromfile(self.filenames[index])

        #one hot encode the labels
        if label is not None:
            #single label
            labels = [0] * len(self.classes)
            labels[self.class_to_idx[label]] = 1
            labels = np.array(labels, dtype=np.float32)
            labels = torch.from_numpy(labels)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            labels = self.target_transform(labels)

        if (self.mode == 'infer') or (labels is None):
            return image
        else:
            return image, labels

    def __len__(self):
        return len(self.filenames)

    def get_filenames(self):
        return self.filenames

    def get_root(self):
        return self.root

    def get_classes(self):
        return self.classes

    def get_class_cardinality(self):
        return self.class_cardinality

    def get_sample_weights(self):
        return self.sample_weights