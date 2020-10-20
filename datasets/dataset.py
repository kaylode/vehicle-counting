import sys
sys.path.append('..')

import os
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as patches

from utils.utils import change_box_order
from augmentations.transforms import Normalize
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_path, transforms=None):

        self.root_dir = root_dir
        self.ann_path = ann_path
        self.transforms = transforms
        self.mode = 'xyxy'
        
        self.coco = COCO(ann_path)
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        box = annot[:, :4]
        label = annot[:, -1]
        
        if self.transforms:
            item = self.transforms(img = img, box = box, label = label)
            img = item['img']
            box = item['box']
            label = item['label']
            
            box = change_box_order(box, order = 'xywh2xyxy')
            
        return {
            'img': img,
            'box': box,
            'label': label
        }

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, image_info['file_name'])
  
        img = Image.open(path).convert('RGB')
    
        return img 

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations
        
        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        return annotations


    def collate_fn(self, batch):
        imgs = torch.stack([s['img'] for s in batch])   
        annots = [torch.cat([s['box'] , s['label'].unsqueeze(1)], dim=1) for s in batch]

        max_num_annots = max(annot.shape[0] for annot in annots)

        if max_num_annots > 0:

            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.ones((len(annots), 1, 5)) * -1

        return {'imgs': imgs, 'labels': annot_padded}


    def visualize_item(self, index = None, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes by index
        """

        if index is None:
            index = random.randint(0,len(self.coco.imgs))
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
            plt.text(x, y-3, self.labels[label], color = color, fontsize=20)
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.show()

    def count_dict(self, types = 1):
        """
        Count class frequencies
        """
        cnt_dict = {}
        if types == 1: # Object Frequencies
            for cl in self.classes.keys():
                num_objs = sum([1 for i in self.coco.anns if self.coco.anns[i]['category_id']-1 == self.classes[cl]])
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

    def __str__(self):
        s = "Custom Dataset for Object Detection\n"
        line = "-------------------------------\n"
        s1 = "Number of samples: " + str(len(self.coco.anns)) + '\n'
        s2 = "Number of classes: " + str(len(self.labels)) + '\n'
        return s + line + s1 + s2

