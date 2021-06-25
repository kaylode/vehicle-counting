import random
import numpy as np
from collections import namedtuple 
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox

class CustomCutout(DualTransform):
    """
    Custom Cutout augmentation with handling of bounding boxes 
    Note: (only supports square cutout regions)
    
    Author: Kaushal28
    Reference: https://arxiv.org/pdf/1708.04552.pdf
    """
    
    def __init__(
        self,
        fill_value=0,
        bbox_removal_threshold=0.50,
        min_cutout_size=192,
        max_cutout_size=512,
        number=1,
        always_apply=False,
        p=0.5
    ):
        """
        Class construstor
        :param fill_value: Value to be filled in cutout (default is 0 or black color)
        :param bbox_removal_threshold: Bboxes having content cut by cutout path more than this threshold will be removed
        :param min_cutout_size: minimum size of cutout (192 x 192)
        :param max_cutout_size: maximum size of cutout (512 x 512)
        """
        super(CustomCutout, self).__init__(always_apply, p)  # Initialize parent class
        self.fill_value = fill_value
        self.bbox_removal_threshold = bbox_removal_threshold
        self.min_cutout_size = min_cutout_size
        self.max_cutout_size = max_cutout_size
        self.number = number
        
    def _get_cutout_position(self, img_height, img_width, cutout_size):
        """
        Randomly generates cutout position as a named tuple
        
        :param img_height: height of the original image
        :param img_width: width of the original image
        :param cutout_size: size of the cutout patch (square)
        :returns position of cutout patch as a named tuple
        """
        position = namedtuple('Point', 'x y')
        return position(
            np.random.randint(0, img_width - cutout_size + 1),
            np.random.randint(0, img_height - cutout_size + 1)
        )
    def _get_cutout(self, img_height, img_width):
        """
        Creates a cutout pacth with given fill value and determines the position in the original image
        
        :param img_height: height of the original image
        :param img_width: width of the original image
        :returns (cutout patch, cutout size, cutout position)
        """
        cutout_size = np.random.randint(self.min_cutout_size, self.max_cutout_size + 1)
        cutout_position = self._get_cutout_position(img_height, img_width, cutout_size)
        return np.full((cutout_size, cutout_size, 3), self.fill_value), cutout_size, cutout_position
    def apply(self, image, **params):
        """
        Applies the cutout augmentation on the given image
        
        :param image: The image to be augmented
        :returns augmented image
        """
        image = image.copy()  # Don't change the original image
        self.img_height, self.img_width, _ = image.shape
        for i in range(self.number):
            cutout_arr, cutout_size, cutout_pos = self._get_cutout(self.img_height, self.img_width)
            
            # Set to instance variables to use this later
            self.image = image
            self.cutout_pos = cutout_pos
            self.cutout_size = cutout_size
            
            image[cutout_pos.y:cutout_pos.y+cutout_size, cutout_pos.x:cutout_size+cutout_pos.x, :] = cutout_arr
        return image
    def apply_to_bbox(self, bbox, **params):
        """
        Removes the bounding boxes which are covered by the applied cutout
        
        :param bbox: A single bounding box coordinates in pascal_voc format
        :returns transformed bbox's coordinates
        """

        # Denormalize the bbox coordinates
        bbox = denormalize_bbox(bbox, self.img_height, self.img_width)
        x_min, y_min, x_max, y_max = tuple(map(int, bbox))
        if x_min >= x_max or y_min >= y_max:
            return normalize_bbox((0, 0, 0, 0), self.img_height, self.img_width)

        bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height
        overlapping_size = np.sum(
            (self.image[y_min:y_max, x_min:x_max, 0] == self.fill_value) &
            (self.image[y_min:y_max, x_min:x_max, 1] == self.fill_value) &
            (self.image[y_min:y_max, x_min:x_max, 2] == self.fill_value)
        )
        # Remove the bbox if it has more than some threshold of content is inside the cutout patch
        if overlapping_size / bbox_size > self.bbox_removal_threshold:
            return normalize_bbox((0, 0, 0, 0), self.img_height, self.img_width)

        return normalize_bbox(bbox, self.img_height, self.img_width)

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return ('fill_value', 'bbox_removal_threshold', 'min_cutout_size', 'max_cutout_size', 'always_apply', 'p')


class CutMix:
    def __init__(self, max_area_drop_pct=0.75) -> None:
        self.max_area_drop_pct = max_area_drop_pct

    def __call__(self, set_images, set_boxes, set_labels, imsize):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize
        s = imsize[0] // 2
    
        xc, yc = [int(random.uniform(h * 0.25, w * 0.75)) for _ in range(2)]  # center x, y

        result_image = np.full((h, w, 3), 1, dtype=np.float32)
        result_boxes = []
        result_labels = np.array([], dtype=np.int)
        
        for i,(image, boxes, labels) in enumerate(zip(set_images, set_boxes, set_labels)):
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
        if self.max_area_drop_pct:
            old_areas = (result_boxes[:,2] - result_boxes[:,0]) * (result_boxes[:,3] -  result_boxes[:,1])

        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        
        if self.max_area_drop_pct:
            new_areas = (result_boxes[:,2] - result_boxes[:,0]) * (result_boxes[:,3] -  result_boxes[:,1])
            
        result_boxes = result_boxes.astype(np.int32)
        index_to_use = np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)

        if self.max_area_drop_pct:
            drop_pct = (old_areas-new_areas)*1.0 / old_areas
            picked_index = np.where(drop_pct < self.max_area_drop_pct)
            index_to_use = np.intersect1d(index_to_use, picked_index)


        result_boxes = result_boxes[index_to_use]
        result_labels = result_labels[index_to_use]
        
        return result_image, result_boxes, result_labels

class MixUp:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(
            self, 
            image, boxes, labels,
            r_image, r_boxes, r_labels):
        """
        Randomly mixes the given list if images with each other
        source: https://www.kaggle.com/kaushal2896/data-augmentation-tutorial-basic-cutout-mixup
        :param images: The images to be mixed up
        :param bboxes: The bounding boxes (labels)
        :param areas: The list of area of all the bboxes
        :param alpha: Required to generate image wieghts (lambda) using beta distribution. In this case we'll use alpha=1, which is same as uniform distribution
        """
        
        # Generate image weight (minimum 0.4 and maximum 0.6)
        lam = np.clip(np.random.beta(self.alpha, self.alpha), 0.4, 0.6)
        
        # Weighted Mixup
        mixedup_images = lam*image + (1 - lam)*r_image

        # Mixup annotations
        mixedup_bboxes = np.vstack((boxes, r_boxes)).astype(np.int32)
        mixedup_labels = np.concatenate((labels, r_labels))

        return mixedup_images, mixedup_bboxes, mixedup_labels