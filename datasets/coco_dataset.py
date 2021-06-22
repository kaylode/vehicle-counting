import sys
sys.path.append('..')

import os
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.postprocess import change_box_order, filter_area
from augmentations.transforms import Denormalize, get_resize_augmentation, get_augmentation
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import albumentations as A
import cv2

class CocoDataset(Dataset):
    def __init__(self, config, root_dir, ann_path, train=True):
        self.root_dir = root_dir
        self.ann_path = ann_path
        self.image_size = config.image_size
        self.ori_image_size = config.image_size
        self.mixup = config.mixup
        self.cutmix = config.cutmix
        self.keep_ratio = config.keep_ratio
        self.multiscale_training = config.multiscale
        self.box_format = 'yxyx' # Output format of the __getitem__
        self.train = train
        
        if self.multiscale_training and train:
            self.init_multiscale_training()
            
        self.resize_transforms = get_resize_augmentation(config.image_size, config.keep_ratio, box_transforms=True)
        self.transforms = get_augmentation(_type="train") if train else get_augmentation(_type="val")

        self.coco = COCO(ann_path)
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def set_box_format(self, format):
        self.box_format = format

    def init_multiscale_training(self):
        self.scale_list = [0.5, 0.75, 1.25, 1]
        self.resize_transforms_list = []
        self.image_size_list = []

        for i in self.scale_list:
            new_image_size = [int(i*self.ori_image_size[0]), int(i*self.ori_image_size[1])]
            self.image_size_list.append(new_image_size)
            self.resize_transforms_list.append(
                get_resize_augmentation(new_image_size, self.keep_ratio, box_transforms=True))

    def set_random_scale(self, random_=False):
        if random_:
            scale = random.choice(range(len(self.scale_list)-1))
            self.resize_transforms = self.resize_transforms_list[scale]
            self.image_size = self.image_size_list[scale]
        else:
            self.resize_transforms = self.resize_transforms_list[-1]
            self.image_size = self.image_size_list[-1]

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.idx_mapping = {}
        self.class_names = []
        for c in categories:
            idx = len(self.classes) + 1
            self.classes[c['name']] = idx
            self.idx_mapping[c['id']] = idx
            self.class_names.append(c['name'])

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.num_classes = len(self.labels)
        
    def __len__(self):
        return len(self.image_ids)

    def load_image_and_boxes(self, idx):
        """
        Load an image and its boxes, also do scaling here
        """
        img, img_name, ori_width, ori_height  = self.load_image(idx)
        img_id = self.image_ids[idx]
        annot = self.load_annotations(idx)
        box = annot[:, :4]
        label = annot[:, -1]
        box = change_box_order(box, order = 'xywh2xyxy')

        if self.resize_transforms is not None:
            resized = self.resize_transforms(
                image=img,
                bboxes=box,
                class_labels=label)
            img = resized['image']
            box = resized['bboxes']
            label = resized['class_labels']

            box = np.array([np.asarray(i) for i in box])
            label = np.array(label)
        if len(box) == 0:
            return self.load_image_and_boxes((idx+1)%len(self.image_ids))

        return img, box, label, img_id, img_name, ori_width, ori_height

    def load_sample(self, idx):
        ori_width, ori_height = None, None
        img_id, img_name = None, None
        if not self.train or random.random() > 0.33:
            image, boxes, labels, img_id, img_name, ori_width, ori_height = self.load_image_and_boxes(idx)
        else:
            if self.mixup and self.cutmix:
                if random.random() > 0.5:
                    image, boxes, labels  = self.load_cutmix_image_and_boxes(idx, self.image_size)
                else:
                    image, boxes, labels = self.load_mixup_image_and_boxes(idx)
            else:
                if self.mixup:
                    image, boxes, labels = self.load_mixup_image_and_boxes(idx)
                elif self.cutmix:
                    image, boxes, labels = self.load_cutmix_image_and_boxes(idx, self.image_size)
                else:
                    image, boxes, labels, img_id, img_name, ori_width, ori_height = self.load_image_and_boxes(idx)

        image = image.astype(np.float32)
        boxes = boxes.astype(np.int32)
        labels = labels.astype(np.int32)

        # Filter small area bboxes
        boxes, labels = filter_area(boxes, labels, min_wh=2, max_wh=4096)

        return image, boxes, labels, img_id, img_name, ori_width, ori_height


    def __getitem__(self, idx):
        
        image, boxes, labels, img_id, img_name, ori_width, ori_height = self.load_sample(idx)
        if self.transforms:
            item = self.transforms(image=image, bboxes=boxes, class_labels=labels)
            # Normalize
            image = item['image']
            boxes = item['bboxes']
            labels = item['class_labels']
            boxes = np.array([np.asarray(i) for i in boxes])
            labels = np.array(labels)

        if len(boxes) == 0:
            return self.__getitem__((idx+1)%len(self.image_ids))
        labels = torch.LongTensor(labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32) 

        target = {}

        if self.box_format == 'yxyx':
            boxes = change_box_order(boxes, 'xyxy2yxyx')

        target['boxes'] = boxes
        target['labels'] = labels
        

        return {
            'img': image,
            'target': target,
            'img_id': img_id,
            'img_name': img_name,
            'ori_size': [ori_width, ori_height]
        }

    def collate_fn(self, batch):
        raise NotImplementedError

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(path)
        height, width, c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        return image, image_info['file_name'], width, height

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
            if a['bbox'][2] <= 2 or a['bbox'][3] <= 2:
                continue
            
            # some annotations have wrong coordinate
            if a['bbox'][0] < 0 or a['bbox'][1] < 0:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox'] # xywh
            annotation[0, 4] = self.idx_mapping[a['category_id']]
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def load_mixup_image_and_boxes(self, index, alpha=1.0):
        """
        Randomly mixes the given list if images with each other
        source: https://www.kaggle.com/kaushal2896/data-augmentation-tutorial-basic-cutout-mixup
        :param images: The images to be mixed up
        :param bboxes: The bounding boxes (labels)
        :param areas: The list of area of all the bboxes
        :param alpha: Required to generate image wieghts (lambda) using beta distribution. In this case we'll use alpha=1, which is same as uniform distribution
        """
        image, boxes, labels, _, _, _, _ = self.load_image_and_boxes(index)    
        r_image, r_boxes, r_labels, _, _, _, _ = self.load_image_and_boxes(random.randint(0, len(self.image_ids) - 1))
        
        # Generate image weight (minimum 0.4 and maximum 0.6)
        lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
        
        # Weighted Mixup
        mixedup_images = lam*image + (1 - lam)*r_image

        # Mixup annotations
        mixedup_bboxes = np.vstack((boxes, r_boxes)).astype(np.int32)
        mixedup_labels = np.concatenate((labels, r_labels))

        return mixedup_images, mixedup_bboxes, mixedup_labels

    def load_cutmix_image_and_boxes(self, index, imsize=[512,512]):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize
        s = imsize[0] // 2
    
        xc, yc = [int(random.uniform(h * 0.25, w * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, len(self.image_ids) - 1) for _ in range(3)]

        result_image = np.full((h, w, 3), 1, dtype=np.float32)
        result_boxes = []
        result_labels = np.array([], dtype=np.int)
        
        current_resize_transforms = self.resize_transforms
        self.resize_transforms = get_resize_augmentation(imsize, keep_ratio=False, box_transforms=True)

        for i,index in enumerate(indexes):
            image, boxes, labels, _, _, _, _ = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            result_labels = np.concatenate((result_labels, labels))
            
        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        index_to_use = np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)
        result_boxes = result_boxes[index_to_use]
        result_labels = result_labels[index_to_use]
        
        self.resize_transforms = current_resize_transforms


        return result_image, result_boxes, result_labels

    def visualize_item(self, index = None, figsize=(15,15)):
        """
        Visualize an image with its bouding boxes by index
        """

        if index is None:
            index = random.randint(0,len(self.coco.imgs))
        item = self.__getitem__(index)
        img_name = item['img_name']
        img = item['img']
        target = item['target']
        box = target['boxes']
        label = target['labels']
        
        normalize = False
        if self.transforms is not None:
            for x in self.transforms.transforms:
                if isinstance(x, A.Normalize):
                    normalize = True
                    denormalize = Denormalize(mean=x.mean, std=x.std)

        # Denormalize and reverse-tensorize
        if normalize:
            img = denormalize(img = img)

        if self.box_format == 'yxyx':
            box = change_box_order(box, 'yxyx2xyxy')

        box = box.numpy()
        label = label.numpy()

        self.visualize(img, box, label, figsize = figsize, img_name= img_name)

    
    def visualize(self, img, boxes, labels, figsize=(15,15), img_name=None):
        """
        Visualize an image with its bouding boxes, input: xyxy
        """
        fig,ax = plt.subplots(figsize=figsize)

        boxes=change_box_order(boxes, 'xyxy2xywh')
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

        if img_name is not None:
            plt.title(img_name)
        plt.show()

    def count_dict(self, types = 1):
        """
        Count class frequencies
        """
        cnt_dict = {}
        if types == 1: # Object Frequencies
            for cl in self.classes.keys():
                num_objs = sum([1 for i in self.coco.anns if self.coco.anns[i]['category_id'] == self.classes[cl]])
                cnt_dict[cl] = num_objs
        elif types == 2:
            widths = [i['width'] for i in self.coco.anns['images']]
            heights = [i['height'] for i in self.coco.anns['images']]
            cnt_dict['height'] = {}
            cnt_dict['width'] = {}
            for i in widths:
                if i not in cnt_dict['width'].keys():
                    cnt_dict['width'][i] = 1
                else:
                    cnt_dict['width'][i] += 1

            for i in heights:
                if i not in cnt_dict['height'].keys():
                    cnt_dict['height'][i] = 1
                else:
                    cnt_dict['height'][i] += 1
        elif types == 3:
            tmp_dict = {}
            for i in self.coco.anns:
                if i['image_id'] not in tmp_dict.keys():
                    tmp_dict[i['image_id']] = 1
                else:
                    tmp_dict[i['image_id']] += 1

            for i in tmp_dict.values():
                if i not in cnt_dict.keys():
                    cnt_dict[i] = 1
                else:
                    cnt_dict[i] += 1
        return cnt_dict

    def plot(self, figsize = (8,8), types = ["freqs"]):
        """
        Plot classes distribution
        """
        ax = plt.figure(figsize = figsize)
        num_plots = len(types)
        plot_idx = 1

        if "freqs" in types:
            ax.add_subplot(num_plots, 1, plot_idx)
            plot_idx +=1
            cnt_dict = self.count_dict(types = 1)
            plt.title("Total objects can be seen: "+ str(sum(list(cnt_dict.values()))))
            bar1 = plt.bar(list(cnt_dict.keys()), list(cnt_dict.values()), color=[np.random.rand(3,) for i in range(len(self.classes))])
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
        
        if "size_freqs" in types:
            ax.add_subplot(num_plots, 1, plot_idx)
            plot_idx +=1
            cnt_dict = self.count_dict(types = 2)
            plt.title("Image sizes distribution: ")
            bar1 = plt.bar(list(cnt_dict['height'].keys()), list(cnt_dict['height'].values()), color='blue')
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
        
        if "object_freqs" in types:
            ax.add_subplot(num_plots, 1, plot_idx)
            plot_idx +=1
            cnt_dict = self.count_dict(types = 3)
            num_objs = sum([i*j for i,j in cnt_dict.items()])
            num_imgs = sum([i for i in cnt_dict.values()])
            mean = num_objs*1.0/num_imgs
            plt.title("Total objects can be seen: "+ str(num_objs) + '\nAverage number object per image: ' + str(np.round(mean, 3)))
          
            bar1 = plt.bar(list(cnt_dict.keys()), list(cnt_dict.values()), color=[np.random.rand(3,) for i in range(len(self.classes))])
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')
            
        plt.show()

    def __str__(self): 
        s1 = "Number of samples: " + str(len(self.coco.anns)) + '\n'
        s2 = "Number of classes: " + str(len(self.labels)) + '\n'
        return s1 + s2
