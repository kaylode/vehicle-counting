import os
import torch
import torch.nn as nn
import torch.utils.data as data
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import numpy as np
from PIL import Image
from augmentations.transforms import Compose, Normalize
from utils.utils import change_box_order

class SegmentationDataset(data.Dataset):
    """
    - Object Detection dataset for COCO Format Only
    - Bounding boxes in format x1,y1,w,h
    - Init argument:
                + img_dir: Directory to images
                + ann_path: Path to annotation file (.json)

    """
    def __init__(self,
                 img_dir,
                 ann_path,
                 transforms = None,
                 max_samples = None,
                 shuffle = False, 
                 mode = 'xywh'):
      
        self.dir = img_dir
        self.ann_path = ann_path
        _, self.ext = os.path.splitext(ann_path)
        self.shuffle = shuffle
        self.transforms = transforms if transforms is not None else Compose()     
        self.max_samples = max_samples
        self.annos = self.load_annos()
        self.labels_to_idx()
        self.fns = self.load_images()
        self.mode =mode

    def labels_to_idx(self):
        """
        Create dictionary for label to indexes
        """
        self.classes_idx = {}
        self.idx_classes = {}
        self.classes = []
        for i in self.annos["categories"]:
            self.classes.append(i['name'])
            self.classes_idx[i['name']] = i['id']
            self.idx_classes[i['id']] = i['name']

    def load_annos(self):
        """
        Read in data from annotations
        """
        with open(self.ann_path, "r") as fi:
            if self.ext == ".json":
                import json
                data = json.load(fi)

        for i in data['annotations']:
            i['category_id'] -= 1               # Label index starts with 0
        for i in data['categories']:
            i['id'] -= 1
        return data

    

    def load_images(self):
        """
        Read in list of paths to images
        """
        data_list = [os.path.join(self.dir,i['file_name']) for i in self.annos['images']]
        if self.shuffle:
            random.shuffle(data_list)
        data_list = data_list[:self.max_samples] if self.max_samples is not None else data_list
        return data_list
    
    def count_dict(self, types = 1):
        """
        Count class frequencies
        """
        cnt_dict = {}
        if types == 1: # Object Frequencies
            for cl in self.classes:
                num_objs = sum([1 for i in self.annos['annotations'] if i['category_id'] == self.classes_idx[cl]])
                cnt_dict[cl] = num_objs
        elif types == 2:
            pass
        return cnt_dict

    def plot(self, figsize = (8,8), types = ["freqs"]):
        """
        Plot classes distribution
        """
        ax = plt.figure(figsize = figsize)
        
        if "freqs" in types:
            cnt_dict = self.count_dict(types = 1)
            plt.title("Total objects can be seen: "+ str(sum(list(cnt_dict.values()))))
            bar1 = plt.bar(list(cnt_dict.keys()), list(cnt_dict.values()), color=[np.random.rand(3,) for i in range(len(self.classes))])
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
        
        plt.show()

    def visualize_item(self, index = None, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes by index
        """

        if index is None:
            index = random.randint(0,len(self.fns))
        item = self.__getitem__(index)
        img = item['img']
        box = item['box']
        label = item['label']

        if any(isinstance(x, Normalize) for x in self.transforms.transforms_list):
            normalize = True
        else:
            normalize = False

        # Denormalize and reverse-tensorize
        if normalize:
            results = self.transforms.denormalize(img = img, box = box, label = label)
            img, label, box = results['img'], results['label'], results['box']
    
        # Numpify
        label = label.numpy()
        box = box.numpy()

        if self.mode == 'xyxy':
            box=change_box_order(box, 'xyxy2xywh')

        self.visualize(img, box, label, figsize = figsize)

    
    def visualize(self, img, boxes, labels, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes
        """
        fig,ax = plt.subplots(figsize=figsize)

        if isinstance(img, torch.Tensor):
            img = img.numpy().squeeze().transpose((1,2,0))
        # Display the image
        ax.imshow(img)

        # Create a Rectangle patch
        for box, label in zip(boxes, labels):
            color = np.random.rand(3,)
            x,y,w,h = box
            rect = patches.Rectangle((x,y),w,h,linewidth=2,edgecolor = color,facecolor='none')
            plt.text(x, y-3, self.idx_classes[label], color = color, fontsize=20)
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.show()

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, index):
        """
        Get an item by index
        """
        img_item = self.annos['images'][index]
        img_id = img_item['id']
        img_name = img_item['file_name']
        img_anno = [i for i in list(self.annos['annotations']) if i['image_id'] == img_id]
        
        img_path = os.path.join(self.dir,img_name)
        box = np.floor(np.array([i['bbox'] for i in img_anno]))
        #mask = [np.array(i['segmentation']).reshape(-1,2) for i in img_anno]

        if self.mode == 'xyxy':
            box=change_box_order(box, 'xywh2xyxy')

        label = np.array([i['category_id'] for i in img_anno]) # Label starts from 0

        img = Image.open(img_path).convert('RGB')

        # Data augmentation
        results = self.transforms(img = img, box = box, label = label)
        
        
        return {
            'img': results['img'],
            'box': results['box'],
            'label': results['label'],
        }

    def collate_fn(self, batch):
        """
        - Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        - This describes how to combine these tensors of different sizes. We use lists.
        - Note: this need not be defined in this Class, can be standalone.
            + param batch: an iterable of N sets from __getitem__()
            + return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
    
        for b in batch:
            images.append(b['img'])
            boxes.append(b['box'])
            labels.append(b['label'])
            
        images = torch.stack(images, dim=0)

        return {
            'imgs': images,
            'boxes': boxes,
            'labels': labels} # tensor (N, 3, 300, 300), 3 lists of N tensors each

    def __str__(self):
        s = "Custom Dataset for Object Detection\n"
        line = "-------------------------------\n"
        s1 = "Number of samples: " + str(len(self.fns)) + '\n'
        s2 = "Number of classes: " + str(len(self.classes)) + '\n'
        return s + line + s1 + s2