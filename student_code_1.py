import math
import random
import glob
import os
import numpy as np

import cv2
import numbers
import collections
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import torch
from torch.utils import data

from utils import resize_image, load_image

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# These are helper functions or functions for demonstration
# You won't need to modify them
#################################################################################


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Compose([
        >>>     Scale(320),
        >>>     RandomSizedCrop(224),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        repr_str = ""
        for t in self.transforms:
            repr_str += t.__repr__() + "\n"
        return repr_str


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly
    (with a probability of 0.5).
    """

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be flipped.

        Returns:
            numpy array: Randomly flipped image
        """
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            return img
        return img

    def __repr__(self):
        return "Random Horizontal Flip"


#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
    """Rescale the input numpy array to the given size.

    This class will resize an input image based on its shortest side.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * height / width)

        interpolations (list of int, optional): Desired interpolation.
            Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
            Pass None during testing: always use CV2.INTER_LINEAR
    """

    def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2
        )
        self.size = size
        # use bilinear if interpolation is not specified
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be scaled.

        Returns:
            numpy array: Rescaled image
        """
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]

        # scale the image
        if isinstance(self.size, int):

            #################################################################################
            new_size = (self.size, int (self.size * img.shape[0] / img.shape[1]) )
            img = cv2.resize(img, new_size)
            # print (new_size)
            # print (img.shape)
            #################################################################################
            return img
        else:
            #################################################################################
            new_size = (img.shape[1], img.shape[0])
            img = cv2.resize(img, new_size)
            print (new_size)
            print (img.shape)
            #################################################################################
            return img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])


class RandomSizedCrop(object):
    """Crop the given numpy array to random area and aspect ratio.

    This class will crop a random region with in an input image. The target area
    / aspect ratio (width/height) of the region is first sampled from a uniform
    distribution. A region satisfying the area / aspect ratio is then sampled
    and cropped. This crop is finally resized to a fixed given size. This is
    widely used as data augmentation for training image classification models.

    Args:
        size (sequence or int): size of target image. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            output size will be (size, size).
        interpolations (list of int, optional): Desired interpolation.
            Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
        area_range (list of int): range of areas to sample from
        ratio_range (list of int): range of aspect ratios to sample from
        num_trials (int): number of sampling trials
    """

    def __init__(
        self,
        size,
        interpolations=_DEFAULT_INTERPOLATIONS,
        area_range=(0.25, 1.0),
        ratio_range=(0.8, 1.2),
        num_trials=10,
    ):
        self.size = size
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations
        self.num_trials = int(num_trials)
        self.area_range = area_range
        self.ratio_range = ratio_range

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]

        for attempt in range(self.num_trials):

            # sample target area / aspect ratio from area range and ratio range
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
            aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

            #################################################################################
            height = int(round((target_area * aspect_ratio) ** 0.5))
            width = int(round((target_area / aspect_ratio) ** 0.5))
            # crop_img = img[self.size[0]-width/2,self.size[0]+width/2 self.size[1]-height/2,self.size[1]+height/2]
            
            # random crop staring coordinates
            if height <= img.shape[0] and width <= img.shape[1]:
                h_start = random.randint(0,img.shape[0] - height)
                w_start = random.randint(0,img.shape[1] - width)
                img = img[h_start : h_start + height, w_start : w_start + width]
                img = Scale(self.size)
            #################################################################################
            # compute the width and height
            # crop the image and resize to output size

        # Fall back
        if isinstance(self.size, int):
            im_scale = Scale(self.size, interpolations=self.interpolations)
            img = im_scale(img)
            #################################################################################
            # Fill in the code here
            # starting point of crop
            h1 = (img.shape[1] - self.size) // 2
            w1 = (img.shape[0] - self.size) // 2
            img = img[h1 : h1 + self.size, w1 : w1 + self.size]
            #################################################################################
            # with a square sized output, the default is to crop the patch in the center
            # (after all trials fail)
            return img
        else:
            # with a pre-specified output size, the default crop is the image itself
            im_scale = Scale(self.size, interpolations=self.interpolations)
            img = im_scale(img)
            return img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return (
            "Random Crop"
            + "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}; Ratio {:.2f} - {:.2f}]".format(
                target_size[0],
                target_size[1],
                self.area_range[0],
                self.area_range[1],
                self.ratio_range[0],
                self.ratio_range[1],
            )
        )


class RandomColor(object):
    """Perturb color channels of a given image.

    This class will apply random color perturbation to an input image. An alpha
    value is first sampled uniformly from the range of (-r, r). 1 + alpha is
    further multiply to a color channel. The sampling is done independently for
    each channel. An efficient implementation can be achieved using a LuT.

    Args:
        color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
    """

    def __init__(self, color_range):
        self.color_range = color_range

    def __call__(self, img):
        #################################################################################
        # Fill in the code here
        #################################################################################
        return img

    def __repr__(self):
        return "Random Color [Range {:.2f} - {:.2f}]".format(
            1 - self.color_range, 1 + self.color_range
        )


class RandomRotate(object):
    """Rotate the given numpy array (around the image center) by a random degree.

    This class will randomly rotate an image and further crop a local region with
    maximum area. A rotation angle is first sampled and then applied to the input.
    A region with maximum area and without any empty pixel is further determined
    and cropped.

    Args:
        degree_range (float): range of degree (-d ~ +d)
    """

    def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
        self.degree_range = degree_range
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]
        # sample rotation
        degree = random.uniform(-self.degree_range, self.degree_range)
        # ignore small rotations
        if np.abs(degree) <= 1.0:
            return img

        #################################################################################
        # Fill in the code here
        h, w = img.shape[:2]
        # Get the image center (about which we will rotate the image)
        center = (w // 2, h // 2)

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, degree, 1.0)

        # Compute the cosine and sine of the rotation matrix (for bounding box calculation)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Compute the new bounding dimensions of the image
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to account for the translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform the rotation
        rotated_img = cv2.warpAffine(img, M, (new_w, new_h), flags=interpolation)

        # Now, crop the largest rectangle from the rotated image
        # Create a mask for the original image size
        mask = np.zeros_like(rotated_img)
        mask = cv2.rectangle(mask, (0, 0), (w, h), (255, 255, 255), -1)
        
        # Apply the mask to the rotated image
        mask_rotated = cv2.warpAffine(mask, M, (new_w, new_h))
        
        # Find the bounding box of the non-zero regions (the area with the actual image)
        x, y, w, h = cv2.boundingRect(mask_rotated[:, :, 0])

        # Crop this region from the rotated image
        img = rotated_img[y:y+h, x:x+w]
        #################################################################################
        # get the rectangular with max area in the rotated image
        return img

    def __repr__(self):
        return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range
        )


#################################################################################
# Additional helper functions. No need to modify.
#################################################################################
class ToTensor(object):
    """Convert a ``numpy.ndarray`` image to tensor.
    Converts a numpy.ndarray (H x W x C) image in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        # convert image to tensor
        assert (img.ndim > 1) and (img.ndim <= 3)
        if img.ndim == 2:
            img = img[:, :, None]
            tensor_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose((2, 0, 1)))
            )
        if img.ndim == 3:
            tensor_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose((2, 0, 1)))
            )
        # backward compatibility
        if isinstance(tensor_img, torch.ByteTensor):
            return tensor_img.float().div(255.0)
        else:
            return tensor_img


class SimpleDataset(data.Dataset):
    """
    A simple dataset using PyTorch dataloader
    """

    def __init__(self, root_folder, file_ext, transforms=None):
        # root folder, split
        self.root_folder = root_folder
        self.transforms = transforms
        self.file_ext = file_ext

        # load all labels
        file_list = glob.glob(os.path.join(root_folder, "*.{:s}".format(file_ext)))
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load image and its label (from file name)
        filename = self.file_list[index]
        img = load_image(filename)
        label = os.path.basename(filename)
        label = label.rstrip(".{:s}".format(self.file_ext))
        # apply data augmentations
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
