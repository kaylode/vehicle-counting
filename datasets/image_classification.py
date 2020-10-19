import torch
import torch.utils.data as data
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from augmentations.transforms import Compose

class ImageClassificationDataset(data.Dataset):
    """
    Reads a folder of images
    """
    def __init__(self,
                img_dir,
                transforms = None,
                max_samples = None,
                shuffle = False):

        self.dir = img_dir
        self.classes = os.listdir(img_dir)
        self.transforms = transforms
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.classes_idx = self.labels_to_idx()
        self.fns = self.load_images()
        

    def labels_to_idx(self):
        indexes = {}
        idx = 0
        for cl in self.classes:
            indexes[cl] = idx
            idx += 1
        return indexes
    
    def load_images(self):
        data_list = []
        for cl in self.classes:
            img_names = sorted(os.listdir(os.path.join(self.dir,cl)))
            for name in img_names:
                data_list.append([cl+'/'+name, cl])
        if self.shuffle:
            random.shuffle(data_list)
        data_list = data_list[:self.max_samples] if self.max_samples is not None else data_list
        return data_list
        
    def __getitem__(self, index):
        img_name, class_name = self.fns[index]
        label = self.classes_idx[class_name]
        img_path = os.path.join(self.dir, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transforms:
            results = self.transforms(img= img, label=[label])
            img = results['img']
            label = results['label']

        return {"img" : img,
                 "label" : label}
    
    def count_dict(self):
        cnt_dict = {}
        for cl in self.classes:
            num_imgs = len(os.listdir(os.path.join(self.dir,cl)))
            cnt_dict[cl] = num_imgs
        return cnt_dict
    

    def visualize_item(self, index = None, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes by index
        """

        if index is None:
            index = np.random.randint(0,len(self.fns))
        item = self.__getitem__(index)
        img = item['img']
        label = item['label']

        # Denormalize and reverse-tensorize
        if any(isinstance(x, Normalize) for x in self.transforms.transforms_list):
            normalize = True
        else:
            normalize = False

        # Denormalize and reverse-tensorize
        if normalize:
            results = self.transforms.denormalize(img = img, box = None, label = label)
            img, label = results['img'], results['label']

        label = label.numpy()
        self.visualize(img, label, figsize = figsize)

    
    def visualize(self, img, label, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes
        """
        fig,ax = plt.subplots(figsize=figsize)

        # Display the image
        ax.imshow(img)
        plt.title(self.classes[label])
        
        plt.show()

    def plot(self, figsize = (8,8), types = ["freqs"]):
        
        ax = plt.figure(figsize = figsize)
        
        if "freqs" in types:
            cnt_dict = self.count_dict()
            plt.title("Classes Distribution")
            bar1 = plt.bar(list(cnt_dict.keys()), list(cnt_dict.values()), color=[np.random.rand(3,) for i in range(len(self.classes))])
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
        
        plt.show()
        
        
    def __len__(self):
        return len(self.fns)
    
    def __str__(self):
        s = "Custom Dataset for Image Classification\n"
        line = "-------------------------------\n"
        s1 = "Number of samples: " + str(len(self.fns)) + '\n'
        s2 = "Number of classes: " + str(len(self.classes)) + '\n'
        return s + line + s1 + s2

    def collate_fn(self, batch):
        """
         - Note: this need not be defined in this Class, can be standalone.
            + param batch: an iterable of N sets from __getitem__()
            + return: a tensor of images, lists of  labels
        """

        images = torch.stack([b['img'] for b in batch], dim=0)
        labels = torch.LongTensor([b['label'] for b in batch])

        return {
            'imgs': images,
            'labels': labels} # tensor (N, 3, 300, 300), 3 lists of N tensors each