import os
import numpy as np
import logging

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image
from skimage.transform import resize

class AIPrimeDataset(Dataset):
    def __init__(self, list_path, img_w=416, img_h=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt'
                            ).replace('.jpg', '.txt') for path in self.img_files]
        self.resized_img_shape = (img_h, img_w) # for resize method use & it requires (h,w)
        self.max_objects = 100

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        #logging.debug(img.shape)
        # Black and white images
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        h, w, _ = img.shape
        #logging.debug(img.shape)

        input_img = resize(img, (*self.resized_img_shape, 3), mode='reflect')
        #logging.debug(input_img.shape)

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------
        # As in C x H x W
        _, resized_h, resized_w = input_img.shape

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            #labels[:, 1] = w * labels[:, 1] / resized_w
            #labels[:, 2] = h * labels[:, 2] / resized_h
            #labels[:, 3] = w * labels[:, 3] / resized_w
            #labels[:, 4] = h * labels[:, 4] / resized_h
        else:
            logging.info("label does not exist, using zero filling: {}".format(label_path))
        # Fill matrix
        # note that this matrix has len=max_objects at dim=0
        filled_labels = np.zeros((self.max_objects, 5), np.float32)
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
