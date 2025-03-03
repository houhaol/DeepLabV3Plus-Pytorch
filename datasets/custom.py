import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

# Convert hex color codes to RGB tuples

class CustomCityscapes:
    # Define the namedtuple structure
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                    'has_instances', 'ignore_in_eval', 'color'])

    # Your label-to-RGB mapping and class details
    LABEL_COLORS_NEW_EN = {
        "#2ca02c": "asphalt",      #0
        "#1f77b4": "concrete",     #1
        "#ff7f0e": "metal",        #2
        "#d62728": "road marking", #3
        "#8c564b": "fabric, leather",#4
        "#7f7f7f": "glass",        #5
        "#bcbd22": "plaster",      #6
        "#ff9896": "plastic",      #7
        "#17becf": "rubber",#8
        "#aec7e8": "sand",         #9
        "#c49c94": "gravel",       #10
        "#c5b0d5": "ceramic",      #11
        "#f7b6d2": "cobblestone",  #12
        "#c7c7c7": "brick",        #13
        "#dbdb8d": "grass",        #14
        "#9edae5": "wood",         #15
        "#393b79": "leaf",         #16
        "#6b6ecf": "water",        #17
        "#9c9ede": "human body",   #18
        "#637939": "sky"           #19
    }

    # Create RGB tuples from the hex codes
    LABEL_COLORS_RGB = {
        index: tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        for index, hex_color in enumerate(LABEL_COLORS_NEW_EN)
    }

    # Define the class details as a list of CityscapesClass
    classes = [
        CityscapesClass('asphalt', 0, 0, 'flat', 1, False, False, LABEL_COLORS_RGB[0]),
        CityscapesClass('concrete', 1, 1, 'flat', 1, False, False, LABEL_COLORS_RGB[1]),
        CityscapesClass('metal', 2, 2, 'flat', 1, False, False, LABEL_COLORS_RGB[2]),
        CityscapesClass('road marking', 3, 3, 'flat', 1, False, False, LABEL_COLORS_RGB[3]),
        CityscapesClass('fabric, leather', 4, 4, 'flat', 1, False, False, LABEL_COLORS_RGB[4]),
        CityscapesClass('glass', 5, 5, 'flat', 1, False, False, LABEL_COLORS_RGB[5]),
        CityscapesClass('plaster', 6, 6, 'flat', 1, False, False, LABEL_COLORS_RGB[6]),
        CityscapesClass('plastic', 7, 7, 'flat', 1, False, False, LABEL_COLORS_RGB[7]),
        CityscapesClass('rubber', 8, 8, 'flat', 1, False, False, LABEL_COLORS_RGB[8]),
        CityscapesClass('sand', 9, 9, 'flat', 1, False, False, LABEL_COLORS_RGB[9]),
        CityscapesClass('gravel', 10, 10, 'flat', 1, False, False, LABEL_COLORS_RGB[10]),
        CityscapesClass('ceramic', 11, 11, 'flat', 1, False, False, LABEL_COLORS_RGB[11]),
        CityscapesClass('cobblestone', 12, 12, 'flat', 1, False, False, LABEL_COLORS_RGB[12]),
        CityscapesClass('brick', 13, 13, 'flat', 1, False, False, LABEL_COLORS_RGB[13]),
        CityscapesClass('grass', 14, 14, 'flat', 1, False, False, LABEL_COLORS_RGB[14]),
        CityscapesClass('wood', 15, 15, 'flat', 1, False, False, LABEL_COLORS_RGB[15]),
        CityscapesClass('leaf', 16, 16, 'nature', 4, False, False, LABEL_COLORS_RGB[16]),
        CityscapesClass('water', 17, 17, 'nature', 4, False, False, LABEL_COLORS_RGB[17]),
        CityscapesClass('human body', 18, 18, 'human', 6, True, False, LABEL_COLORS_RGB[18]),
        CityscapesClass('sky', 19, 19, 'sky', 5, False, False, LABEL_COLORS_RGB[19]),
        CityscapesClass('background', -1, 255, 'void', 0, False, True, (0, 0, 0))
    ]
    
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])


    def __init__(self, root, split='train', mode='gtFine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = mode  # 'gtFine' or 'gtCoarse'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                            ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                            ' specified "split" and "mode" are inside the "root" directory')
        
        # Iterate through the city subdirectories in images
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            if not os.path.isdir(img_dir) or not os.path.isdir(target_dir):
                continue
            
            # Iterate through the image files in each city
            for file_name in os.listdir(img_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    self.images.append(os.path.join(img_dir, file_name))
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                self._get_target_suffix(self.mode, self.target_type))
                    self.targets.append(os.path.join(target_dir, target_name))


    @classmethod
    def encode_target(cls, target):
        target[target == 255] = 20
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 20
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)
