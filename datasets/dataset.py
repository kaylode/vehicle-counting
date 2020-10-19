import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', types='train', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        # 
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + types + '.json'))
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
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
  
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

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

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


    def collate_fn(self, data):
        imgs = [s['img'] for s in data]
        annots = [s['annot'] for s in data]
        scales = [s['scale'] for s in data]

        imgs = torch.from_numpy(np.stack(imgs, axis=0))

        max_num_annots = max(annot.shape[0] for annot in annots)

        if max_num_annots > 0:

            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.ones((len(annots), 1, 5)) * -1

        imgs = imgs.permute(0, 3, 1, 2)

        return {'imgs': imgs, 'labels': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}

class Rotation(object):
    def __init__(self, angle = 10):
        self.angle= angle
    def __call__(self, sample):
        '''
            Rotate image and bounding box
            image: A Pil image (w, h)
            boxes: A tensors of dimensions (#objects, 4)
            
            Out: rotated image (w, h), rotated boxes
        '''

         
        image, boxes = sample['img'], sample['annot']
        

        new_image = image.copy()
        new_boxes = boxes.copy()
        
        #Rotate image, expand = True
        w = image.shape[1]
        h = image.shape[0]
        cx = w/2
        cy = h/2
        new_image = new_image.rotate(self.angle, expand= True)
        angle = np.radians(self.angle)
        alpha = np.cos(angle)
        beta = np.sin(angle)
        #Get affine matrix
        AffineMatrix = torch.tensor([[alpha, beta, (1-alpha)*cx - beta*cy],
                                    [-beta, alpha, beta*cx + (1-alpha)*cy]])
        
        #Rotation boxes
        box_width = (boxes[:,2] - boxes[:,0]).reshape(-1,1)
        box_height = (boxes[:,3] - boxes[:,1]).reshape(-1,1)
        
        #Get corners for boxes
        x1 = boxes[:,0].reshape(-1,1)
        y1 = boxes[:,1].reshape(-1,1)
        
        x2 = x1 + box_width
        y2 = y1 
        
        x3 = x1
        y3 = y1 + box_height
        
        x4 = boxes[:,2].reshape(-1,1)
        y4 = boxes[:,3].reshape(-1,1)
        
        corners = torch.stack((x1,y1,x2,y2,x3,y3,x4,y4), dim= 1)
        corners.reshape(8, 8)    #Tensors of dimensions (#objects, 8)
        corners = corners.reshape(-1,2) #Tensors of dimension (4* #objects, 2)
        corners = torch.cat((corners, torch.ones(corners.shape[0], 1)), dim= 1) #(Tensors of dimension (4* #objects, 3))
        
        cos = np.abs(AffineMatrix[0, 0])
        sin = np.abs(AffineMatrix[0, 1])
        
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        AffineMatrix[0, 2] += (nW / 2) - cx
        AffineMatrix[1, 2] += (nH / 2) - cy
        
        #Apply affine transform
        rotate_corners = torch.mm(AffineMatrix, corners.t()).t()
        rotate_corners = rotate_corners.reshape(-1,8)
        
        x_corners = rotate_corners[:,[0,2,4,6]]
        y_corners = rotate_corners[:,[1,3,5,7]]
        
        #Get (x_min, y_min, x_max, y_max)
        x_min, _ = torch.min(x_corners, dim= 1)
        x_min = x_min.reshape(-1, 1)
        y_min, _ = torch.min(y_corners, dim= 1)
        y_min = y_min.reshape(-1, 1)
        x_max, _ = torch.max(x_corners, dim= 1)
        x_max = x_max.reshape(-1, 1)
        y_max, _ = torch.max(y_corners, dim= 1)
        y_max = y_max.reshape(-1, 1)
        
        new_boxes = torch.cat((x_min, y_min, x_max, y_max), dim= 1)
        
        scale_x = new_image.width / w
        scale_y = new_image.height / h
        
        #Resize new image to (w, h)
        new_image = new_image.resize((500, 333))
        
        #Resize boxes
        new_boxes /= torch.Tensor([scale_x, scale_y, scale_x, scale_y])
        new_boxes[:, 0] = torch.clamp(new_boxes[:, 0], 0, w)
        new_boxes[:, 1] = torch.clamp(new_boxes[:, 1], 0, h)
        new_boxes[:, 2] = torch.clamp(new_boxes[:, 2], 0, w)
        new_boxes[:, 3] = torch.clamp(new_boxes[:, 3], 0, h)

        return {'img': new_image, 
                'annot': new_boxes}

class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
