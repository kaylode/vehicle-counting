import torchvision.transforms.functional as TF
import random
import numpy as np
import torch
import torchvision
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image
import cv2
from utils.utils import change_box_order, find_intersection, find_jaccard_overlap

class Normalize(object):
        """
        Mean and standard deviation of ImageNet data
        :param mean: (list of float)
        :param std: (list of float)
        """
        def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],box_transform = True, **kwargs):
            self.mean = mean
            self.std = std
            self.box_transform = box_transform
        def __call__(self, img, box = None, label = None, mask = None, **kwargs):
            """
            :param img: (tensor) image to be normalized
            :param box: (list of tensor) bounding boxes to be normalized, by dividing them with image's width and heights. Format: (x,y,width,height)
            """
            new_img = TF.normalize(img, mean = self.mean, std = self.std)
            if box is not None and self.box_transform:
                _, i_h, i_w = img.size()
                for bb in box:
                    bb[0] = bb[0]*1.0 / i_w
                    bb[1] = bb[1]*1.0 / i_h
                    bb[2] = bb[2]*1.0 / i_w
                    bb[3] = bb[3]*1.0 / i_h

            results = {
                'img': new_img,
                'box': box,
                'label': label,
                'mask': mask}
    
            return results

class ToPILImage(object):
    """Convert a tensor or an ndarray to PIL Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
             - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
             - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
             - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
             - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
               ``short``).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, img, box = None, label = None, mask = None, **kwargs):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray((img*255).astype(np.uint8))
        elif isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img, self.mode)
        return {
                'img': img,
                'box': box,
                'label': label,
                'mask': mask
            }

class Denormalize(object):
        """
        Denormalize image and boxes for visualization
        """
        def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],box_transform=True, **kwargs):
            self.mean = mean
            self.std = std
            self.box_transform = box_transform
        def __call__(self, img, box = None, label = None, mask = None, **kwargs):
            """
            :param img: (tensor) image to be denormalized
            :param box: (list of tensor) bounding boxes to be denormalized, by multiplying them with image's width and heights. Format: (x,y,width,height)
            """
            mean = np.array(self.mean)
            std = np.array(self.std)
            img_show = img.numpy().squeeze().transpose((1,2,0))
            img_show = (img_show * std+mean)
            img_show = np.clip(img_show,0,1)


            if box is not None and self.box_transform:
                _, i_h, i_w = img.size()
                for bb in box:
                    bb[0] = bb[0]* i_w
                    bb[1] = bb[1]* i_h
                    bb[2] = bb[2]* i_w
                    bb[3] = bb[3]* i_h

            results = {
                'img': img_show,
                'box': box,
                'label': label,
                'mask': mask}
    
            return results
           
class ToTensor(object):
        """
        Tensorize image
        """
        def __init__(self):
            pass
        def __call__(self, img, box = None, label=None,  mask = None, **kwargs):
            """
            :param img: (PIL Image) image to be tensorized
            :param box: (list of float) bounding boxes to be tensorized. Format: (x,y,width,height)
            :param label: (int) bounding boxes to be tensorized. Format: (x,y,width,height)
            """

            img = TF.to_tensor(img)
            
            if label is not None:
                label = torch.LongTensor(label)
            if box is not None:
                box = torch.as_tensor(box, dtype=torch.float32)             
            if mask is not None:
                mask = np.array(mask)
                mask = torch.from_numpy(mask).long()

            return {
                'img': img,
                'box': box,
                'label': label,
                'mask': mask}
           
class Resize(object):
        """
        - Resize an image and bounding box, mask
        - Argument:
                    + img: PIL Image
                    + box: list of bounding box for each objects in the image
                    + size: image new size
        """
        def __init__(self, size = (224,224), **kwargs):
            #size: width,height
            self.size = size

        def __call__(self, img, label=None, box = None,  mask = None, **kwargs):
            # Resize image
            new_img = img.resize(self.size, Image.BILINEAR)
        
            
            if box is not None:
                np_box = np.array(box)
                old_dims = np.array([img.width, img.height, img.width, img.height])
                new_dims = np.array([self.size[0], self.size[1], self.size[0], self.size[1]])

                # Resize bounding box and round down
                box = np.floor((np_box / old_dims) * new_dims)
            
            if mask is not None:
                mask = mask.resize(self.size, Image.NEAREST)
            
            results = {
                'img': new_img,
                'box': box,
                'label': label,
                'mask': mask}
    
            return results

class Rotation(object):
    '''
        Source: https://github.com/Paperspace/DataAugmentationForObjectDetection
        Rotate image and bounding box
        - Argument:
                    + img: PIL Image
                    + box: list of bounding box for each objects in the image
                    + size: image new size
    '''
    def __init__(self, angle=10):
        self.angle = angle
        if not type(self.angle) == tuple:
            self.angle = (-self.angle, self.angle)


    def rotate_im(self,image, angle):
        """Rotate the image.
        
        Rotate the image such that the rotated image is enclosed inside the tightest
        rectangle. The area not occupied by the pixels of the original image is colored
        black. 
        
        Parameters
        ----------
        
        image : numpy.ndarray
            numpy image
        
        angle : float
            angle by which the image is to be rotated
        
        Returns
        -------
        
        numpy.ndarray
            Rotated Image
        
        """
        # grab the dimensions of the image and then determine the
        # centre
        (h, w) = image.height, image.width
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        image = cv2.warpAffine(np.array(image), M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
        return image


    def bbox_area(self,bbox):
        return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])
    
    def get_corners(self, bboxes):
        """Get corners of bounding boxes

        Parameters
        ----------

        bboxes: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`

        returns
        -------

        numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      

        """
        width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
        height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)

        x1 = bboxes[:,0].reshape(-1,1)
        y1 = bboxes[:,1].reshape(-1,1)

        x2 = x1 + width
        y2 = y1 

        x3 = x1
        y3 = y1 + height

        x4 = bboxes[:,2].reshape(-1,1)
        y4 = bboxes[:,3].reshape(-1,1)

        corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))

        return corners

    def rotate_box(self, corners,angle,  cx, cy, h, w):
        """Rotate the bounding box.
        
        
        Parameters
        ----------
        
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        
        angle : float
            angle by which the image is to be rotated
            
        cx : int
            x coordinate of the center of image (about which the box will be rotated)
            
        cy : int
            y coordinate of the center of image (about which the box will be rotated)
            
        h : int 
            height of the image
            
        w : int 
            width of the image
        
        Returns
        -------
        
        numpy.ndarray
            Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        """

        corners = corners.reshape(-1,2)
        corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
        
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        calculated = np.dot(M,corners.T).T
        
        calculated = calculated.reshape(-1,8)
        
        return calculated

    def get_enclosing_box(self, corners):
        """Get an enclosing box for ratated corners of a bounding box
        
        Parameters
        ----------
        
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
        
        Returns 
        -------
        
        numpy.ndarray
            Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
            
        """
        x_ = corners[:,[0,2,4,6]]
        y_ = corners[:,[1,3,5,7]]
        
        xmin = np.min(x_,1).reshape(-1,1)
        ymin = np.min(y_,1).reshape(-1,1)
        xmax = np.max(x_,1).reshape(-1,1)
        ymax = np.max(y_,1).reshape(-1,1)
        
        final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
        
        return final

    def clip_box(self, bbox, clip_box, alpha):
        """Clip the bounding boxes to the borders of an image

        Parameters
        ----------

        bbox: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`

        clip_box: numpy.ndarray
            An array of shape (4,) specifying the diagonal co-ordinates of the image
            The coordinates are represented in the format `x1 y1 x2 y2`

        alpha: float
            If the fraction of a bounding box left in the image after being clipped is 
            less than `alpha` the bounding box is dropped. 

        Returns
        -------

        numpy.ndarray
            Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes left are being clipped and the bounding boxes are represented in the
            format `x1 y1 x2 y2` 

        """
        ar_ = (self.bbox_area(bbox))
        x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
        y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
        x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
        y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)

        bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))

        delta_area = ((ar_ - self.bbox_area(bbox))/ar_)

        mask = (delta_area < (1 - alpha)).astype(int)

        bbox = bbox[mask == 1,:]
        return bbox

    def __call__(self, img, box = None, label = None, mask = None, **kwargs):
        
        
        angle = random.uniform(*self.angle)
        w,h = img.width, img.height
        cx, cy = w//2, h//2
        img = self.rotate_im(img, angle)
        
        if mask is not None:
            mask = self.rotate_im(mask, angle)
            

        if box is not None:
            new_box = change_box_order(box, 'xywh2xyxy')
            corners = self.get_corners(new_box)
            corners = np.hstack((corners, new_box[:,4:]))
            corners[:,:8] = self.rotate_box(corners[:,:8], angle, cx, cy, h, w)
            new_bbox = self.get_enclosing_box(corners)
            scale_factor_x = img.shape[1] / w
            scale_factor_y = img.shape[0] / h
            img = cv2.resize(img, (w,h))
            if mask is not None:
                mask = cv2.resize(mask, (w,h))
            new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
            new_box = new_bbox
            new_box = self.clip_box(new_box, [0,0,w, h], 0.25)
            new_box = change_box_order(new_box, 'xyxy2xywh')
        else:
            new_box = box
        
        img = Image.fromarray(img)
        mask = Image.fromarray(mask) if mask is not None else None

        return {
            'img': img, 
            'box': new_box,
            'label': label,
            'mask': mask}
          
class RandomShear(object):
    """Randomly shears an image in horizontal direction   
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    shear_factor: float or tuple(float)
        if **float**, the image is sheared horizontally by a factor drawn 
        randomly from a range (-`shear_factor`, `shear_factor`). If **tuple**,
        the `shear_factor` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, shear_factor = 0.2):
        self.shear_factor = shear_factor
        
        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for scaling factor"   
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)
        
        shear_factor = random.uniform(*self.shear_factor)
        
    def __call__(self, img, box = None, label = None, mask = None, **kwargs):
    
        shear_factor = random.uniform(*self.shear_factor)
        img = np.array(img)
        mask = np.array(mask) if mask is not None else None
        w,h = img.shape[1], img.shape[0]
    
        if shear_factor < 0:
          if mask is not None:
            mask = Image.fromarray(mask)
          img = Image.fromarray(img)
          item = RandomHorizontalFlip(1)(img = img, box = box, mask = mask)
          img, box = item['img'], item['box']
          img = np.array(img)
          if mask is not None:
            mask = item['mask']
            mask = np.array(mask)
          
    
        M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
        nW =  img.shape[1] + abs(shear_factor*img.shape[0])

        if box is not None:
          box = change_box_order(box, 'xywh2xyxy')
          box[:,[0,2]] += ((box[:,[1,3]]) * abs(shear_factor) ).astype(int) 
          box = change_box_order(box, 'xyxy2xywh')
    
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
        if mask is not None:
          mask = cv2.warpAffine(mask, M, (int(nW), mask.shape[0]))

        if shear_factor < 0:
          if mask is not None:
            mask = Image.fromarray(mask)
          img = Image.fromarray(img)
          item = RandomHorizontalFlip(1)(img = img, box = box, mask = mask)
          img, box = item['img'], item['box']
          img = np.array(img)
          if mask is not None:
            mask = item['mask']
            mask = np.array(mask)
    
        img = cv2.resize(img, (w,h))
        mask = cv2.resize(mask, (w,h)) if mask is not None else None

        scale_factor_x = nW / w

        if box is not None:
          box = change_box_order(box, 'xywh2xyxy')
          box[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1] 
          box = change_box_order(box, 'xyxy2xywh')

        img = Image.fromarray(img)
        mask = Image.fromarray(mask) if mask is not None else None
        return {
            'img': img, 
            'box': box,
            'label': label,
            'mask': mask}

class RandomVerticalFlip(object):
        """
        Horizontally flip image and its bounding box, mask
        """
        def __init__(self, ratio = 0.5):
            self.ratio = ratio
          
        def __call__(self, img, box = None, label=None,  mask = None, **kwargs):
            if random.randint(1,10) <= self.ratio*10:
                # Flip image
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
                # Flip mask
                if mask is not None:
                    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                    
                # Flip bounding box
                if box is not None:
                    new_box = change_box_order(box, 'xywh2xyxy')
                    h = img.width
                    ymin = h - new_box[:,3]
                    ymax = h - new_box[:,1]
                    new_box[:,1] = ymin
                    new_box[:,3] = ymax
                    new_box = change_box_order(new_box, 'xyxy2xywh')
                    box = new_box
            

            results = {
                'img': img,
                'box': box,
                'label': label,
                'mask': mask}
    
            return results

class RandomHorizontalFlip(object):
        """
        Horizontally flip image and its bounding box, mask
        """
        def __init__(self, ratio = 0.5):
            self.ratio = ratio
          
        def __call__(self, img, box = None, label=None,  mask = None, **kwargs):
            if random.randint(1,10) <= self.ratio*10:
                # Flip image
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
                # Flip mask
                if mask is not None:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                    
                # Flip bounding box
                if box is not None:
                    new_box = change_box_order(box, 'xywh2xyxy')
                    w = img.width
                    xmin = w - new_box[:,2]
                    xmax = w - new_box[:,0]
                    new_box[:,0] = xmin
                    new_box[:,2] = xmax
                    new_box = change_box_order(new_box, 'xyxy2xywh')
                    box = new_box
            

            results = {
                'img': img,
                'box': box,
                'label': label,
                'mask': mask}
    
            return results

class RandomCrop(object):
    """
    Source: https://github.com/anhtuan85/Data-Augmentation-for-Object-Detection
    """
    def __init__(self):
        self.ratios = [0.3, 0.5, 0.7, 0.9, None]
        
    def __call__(self, img, box = None, label = None, mask = None, **kwargs):
        '''
        image: A PIL image
        boxes: Bounding boxes, a tensor of dimensions (#objects, 4)
        labels: labels of object, a tensor of dimensions (#objects)
        difficulties: difficulties of detect object, a tensor of dimensions (#objects)
        
        Out: cropped image , new boxes, new labels, new difficulties
        '''
      
        image = TF.to_tensor(img)
        masks = TF.to_tensor(mask) if mask is not None else mask
        original_h = image.size(1)
        original_w = image.size(2)

        while True:
            mode = random.choice(self.ratios)

            if mode is None:
                return {
                    'img': img,
                    'box': box,
                    'label': label,
                    'mask': mask}

            if box is not None:
                boxes = change_box_order(box, 'xywh2xyxy')
                boxes = torch.FloatTensor(boxes)
                labels = torch.LongTensor(label)
            else:
                boxes = None
                labels = None
            
                
            new_image = image
            new_boxes = boxes
            new_labels = labels
            new_mask = masks if mask is not None else mask

            for _ in range(50):
                # Crop dimensions: [0.3, 1] of original dimensions
                new_h = random.uniform(0.3*original_h, original_h)
                new_w = random.uniform(0.3*original_w, original_w)

                # Aspect ratio constraint b/t .5 & 2
                if new_h/new_w < 0.5 or new_h/new_w > 2:
                    continue

                #Crop coordinate
                left = random.uniform(0, original_w - new_w)
                right = left + new_w
                top = random.uniform(0, original_h - new_h)
                bottom = top + new_h
                crop = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])

                # Calculate IoU  between the crop and the bounding boxes
                if boxes is not None:
                    overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes) #(1, #objects)
                    overlap = overlap.squeeze(0)
                    # If not a single bounding box has a IoU of greater than the minimum, try again
                    if overlap.max().item() < mode:
                        continue

                #Crop
                new_image = image[:, int(top):int(bottom), int(left):int(right)] #(3, new_h, new_w)
                new_masks = masks[:, int(top):int(bottom), int(left):int(right)] if masks is not None else masks

                #Center of bounding boxes
                if boxes is not None:
                    center_bb = (boxes[:, :2] + boxes[:, 2:])/2.0

                    #Find bounding box has been had center in crop
                    center_in_crop = (center_bb[:, 0] >left) * (center_bb[:, 0] < right
                                    ) *(center_bb[:, 1] > top) * (center_bb[:, 1] < bottom)    #( #objects)

                    if not center_in_crop.any():
                        continue

                    #take matching bounding box
                    new_boxes = boxes[center_in_crop, :]

                    #take matching labels
                    new_labels = labels[center_in_crop]

                    #Use the box left and top corner or the crop's
                    new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])

                    #adjust to crop
                    new_boxes[:, :2] -= crop[:2]

                    new_boxes[:, 2:] = torch.min(new_boxes[:, 2:],crop[2:])

                    #adjust to crop
                    new_boxes[:, 2:] -= crop[:2]
                
                    new_boxes = change_box_order(new_boxes, 'xyxy2xywh')
                    new_boxes = new_boxes.numpy()
                    new_labels = new_labels.numpy()
                else:
                    new_boxes = None

                new_masks = TF.to_pil_image(new_masks) if new_masks is not None else None

                return {
                        'img': TF.to_pil_image(new_image),
                        'box': new_boxes,
                        'label': new_labels,
                        'mask': new_masks}

class ColorJitter(object):
        """
        Jit the image's color
        :param brightness (float or tuple of float (min, max)): How much to jitter brightness.  
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]  
            or the given [min, max]. Should be non negative numbers.  
        :param contrast (float or tuple of float (min, max)): How much to jitter contrast.  
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]  
            or the given [min, max]. Should be non negative numbers.  
        :param saturation (float or tuple of float (min, max)): How much to jitter saturation.  
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]  
            or the given [min, max]. Should be non negative numbers.  
        :param hue (float or tuple of float (min, max)): How much to jitter hue.  
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].  
            Should have 0 = hue  = 0.5 or -0.5  = min  = max  = 0.5. 
        """
        def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):

            self.jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

        def __call__(self, img, box = None, label = None, mask = None, **kwargs):
            return {
                'img': self.jitter(img),
                'box': box,
                'label': label,
                'mask': mask
            }

class Cutout(object):
    #https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, holes_ratio=0.05, length=8):
        self.holes_ratio = holes_ratio
        self.length = length

    def __call__(self, img, box = None, label = None, mask = None, **kwargs):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        
        n_holes = int( h*w* self.holes_ratio / (self.length *self.length) )
        
        mask = np.ones((h, w), np.float32)

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return {
                'img': img,
                'box': box,
                'label': label,
                'mask': mask
            }

class RandAugment(object):
    #https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
    def ShearX(self, img, v):  # [-0.3, 0.3]
        assert -0.3 <= v <= 0.3
        if random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

    def ShearY(self, img, v):  # [-0.3, 0.3]
        assert -0.3 <= v <= 0.3
        if random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


    def TranslateX(self, img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
        assert -0.45 <= v <= 0.45
        if random.random() > 0.5:
            v = -v
        v = v * img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


    def TranslateXabs(self, img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
        assert 0 <= v
        if random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


    def TranslateY(self, img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
        assert -0.45 <= v <= 0.45
        if random.random() > 0.5:
            v = -v
        v = v * img.size[1]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


    def TranslateYabs(self, img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
        assert 0 <= v
        if random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


    def Rotate(self, img, v):  # [-30, 30]
        assert -30 <= v <= 30
        if random.random() > 0.5:
            v = -v
        return img.rotate(v)


    def AutoContrast(self, img, _):
        return PIL.ImageOps.autocontrast(img)


    def Invert(self, img, _):
        return PIL.ImageOps.invert(img)


    def Equalize(self, img, _):
        return PIL.ImageOps.equalize(img)


    def Flip(self, img, _):  # not from the paper
        return PIL.ImageOps.mirror(img)


    def Solarize(self, img, v):  # [0, 256]
        assert 0 <= v <= 256
        return PIL.ImageOps.solarize(img, v)


    def SolarizeAdd(self, img, addition=0, threshold=128):
        img_np = np.array(img).astype(np.int)
        img_np = img_np + addition
        img_np = np.clip(img_np, 0, 255)
        img_np = img_np.astype(np.uint8)
        img = Image.fromarray(img_np)
        return PIL.ImageOps.solarize(img, threshold)


    def Posterize(self, img, v):  # [4, 8]
        v = int(v)
        v = max(1, v)
        return PIL.ImageOps.posterize(img, v)


    def Contrast(self, img, v):  # [0.1,1.9]
        assert 0.1 <= v <= 1.9
        return PIL.ImageEnhance.Contrast(img).enhance(v)


    def Color(self, img, v):  # [0.1,1.9]
        assert 0.1 <= v <= 1.9
        return PIL.ImageEnhance.Color(img).enhance(v)


    def Brightness(self, img, v):  # [0.1,1.9]
        assert 0.1 <= v <= 1.9
        return PIL.ImageEnhance.Brightness(img).enhance(v)


    def Sharpness(self, img, v):  # [0.1,1.9]
        assert 0.1 <= v <= 1.9
        return PIL.ImageEnhance.Sharpness(img).enhance(v)


    def Cutout(self, img, v):  # [0, 60] => percentage: [0, 0.2]
        assert 0.0 <= v <= 0.2
        if v <= 0.:
            return img

        v = v * img.size[0]
        return CutoutAbs(img, v)


    def CutoutAbs(self, img, v):  # [0, 60] => percentage: [0, 0.2]
        # assert 0 <= v <= 20
        if v < 0:
            return img
        w, h = img.size
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)

        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = min(w, x0 + v)
        y1 = min(h, y0 + v)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        # color = (0, 0, 0)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img


    def SamplePairing(self, imgs):  # [0, 0.4]
        def f(img1, v):
            i = np.random.choice(len(imgs))
            img2 = PIL.Image.fromarray(imgs[i])
            return PIL.Image.blend(img1, img2, v)

        return f


    def Identity(self, img, v):
        return img


    def augment_list(self):  # 16 oeprations and their ranges
        #https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
        l = [
            (self.Identity, 0., 1.0),
            (self.ShearX, 0., 0.3),  # 0
            (self.ShearY, 0., 0.3),  # 1
            (self.TranslateX, 0., 0.33),  # 2
            (self.TranslateY, 0., 0.33),  # 3
            (self.Rotate, 0, 30),  # 4
            (self.AutoContrast, 0, 1),  # 5
            (self.Invert, 0, 1),  # 6
            (self.Equalize, 0, 1),  # 7
            (self.Solarize, 0, 110),  # 8
            (self.Posterize, 4, 8),  # 9
            # (Contrast, 0.1, 1.9),  # 10
            (self.Color, 0.1, 1.9),  # 11
            (self.Brightness, 0.1, 1.9),  # 12
            (self.Sharpness, 0.1, 1.9),  # 13
            # (Cutout, 0, 0.2),  # 14
            # (SamplePairing(imgs), 0, 0.4),  # 15
        ]

        # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
        # l = [
        #     (self.AutoContrast, 0, 1),
        #     (self.Equalize, 0, 1),
        #     (self.Invert, 0, 1),
        #     (self.Rotate, 0, 30),
        #     (self.Posterize, 0, 4),
        #     (self.Solarize, 0, 256),
        #     (self.SolarizeAdd, 0, 110),
        #     (self.Color, 0.1, 1.9),
        #     (self.Contrast, 0.1, 1.9),
        #     (self.Brightness, 0.1, 1.9),
        #     (self.Sharpness, 0.1, 1.9),
        #     (self.ShearX, 0., 0.3),
        #     (self.ShearY, 0., 0.3),
        #     (self.CutoutAbs, 0, 40),
        #     (self.TranslateXabs, 0., 100),
        #     (self.TranslateYabs, 0., 100),
        # ]

        return l

    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = self.augment_list()

    def __call__(self, img, box = None, label = None, mask = None, **kwargs):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return {
                'img': img,
                'box': box,
                'label': label,
                'mask': mask
            }

def do_nothing(img, box = None, label = None, mask = None, **kwargs):
    return {
        'img': img,
        'mask': mask, 
        'box': box,
        'label': label
    }

def enable_if(condition, obj):
    return obj if condition else do_nothing

def get_augmentation(config, types = 'train'):
    return Compose([
        Resize(size = config.image_size),
        enable_if(config.augmentations.get('horizontal_flip', 0) > 0 and types == 'train', RandomHorizontalFlip(config.augmentations['horizontal_flip'])),
        enable_if(config.augmentations.get('shear', 0) > 0 and types == 'train', RandomShear(config.augmentations['shear'])),
        enable_if(config.augmentations.get('rotation', 0) > 0 and types == 'train', Rotation(config.augmentations['rotation'])),
        enable_if(config.augmentations.get('colorjitter') is not None and types == 'train', ColorJitter(**config.augmentations['colorjitter'])),
        ToTensor(),
        enable_if(config.augmentations.get('cutout', 0) > 0 and types == 'train', Cutout(config.augmentations['cutout'])),
        Normalize(box_transform=False)
    ])

class Compose(object):
        """
        - Custom Transform class include image augmentation methods, return dict
        - Can apply for all tasks
        - Examples:
                    my_transforms = Compose(transforms_list=[
                                                Resize((300,300)),
                                                #RandomHorizontalFlip(),
                                                ToTensor(),
                                                Normalize()])
                    results = my_transforms(img, box, label)
                    img, box, label, mask = results['img'], results['box'], results['label'], results['mask']
        """
        def __init__(self, transforms_list = None):
            
            
            if transforms_list is None:
                self.transforms_list = [Resize(), ToTensor(), Normalize()]
            else:
              self.transforms_list = transforms_list
            if not isinstance(self.transforms_list,list):
                self.transforms_list = list(self.transforms_list)
            
            for x in self.transforms_list:
                if isinstance(x, Normalize):
                    self.denormalize = Denormalize(box_transform=x.box_transform)

        def __call__(self, img, box = None, label = None, mask = None):
            for tf_ in self.transforms_list:
                results = tf_(img = img, box = box, label = label, mask = mask)
                img = results['img']
                box = results['box']
                label = results['label']
                mask = results['mask']

            return results