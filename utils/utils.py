import torch
import torch.nn as nn
import torchvision
import numpy as np

def one_hot_embedding(labels, num_classes):
    '''
    Embedding labels to one-hot form.

    :param labels: (LongTensor) class labels, sized [N,].
    :param num_classes: (int) number of classes.
    :return: (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


def box_nms(boxes, scores, threshold=0.5):
    """
    Non Maximum Suppression
    Use custom (very slow) or torchvision non-maximum supression on bounding boxes
    
    :param bboxes: (tensor) bounding boxes, size [N, 4]
    :param scores: (tensor) bbox scores, sized [N]
    :return: keep: (tensor) selected box's indices
    """

    # Torchvision NMS:
    keep = torchvision.ops.boxes.nms(boxes, scores,threshold)
    return keep

    # Custom NMS: uncomment to use
    """x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        try:
            i = order[0]
        except IndexError:
            break
        keep.append(i)

        if order.numel() == 1:
            break
        
        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr < threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        # because the length of the ovr is less than the order by 1
        # so we have to add to ids to get the right one
        order = order[ids + 1]
    return torch.LongTensor(keep)"""

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def change_box_order(boxes, order):
    """
    Change box order between (xmin, ymin, xmax, ymax) and (xcenter, ycenter, width, height).

    :param boxes: (tensor) or {np.array) bounding boxes, sized [N, 4]
    :param order: (str) ['xyxy2xywh', 'xywh2xyxy', 'xyxy2cxcy', 'cxcy2xyxy']
    :return: (tensor) converted bounding boxes, size [N, 4]
    """

    assert order in ['xyxy2xywh', 'xywh2xyxy', 'xyxy2cxcy', 'cxcy2xyxy']

    # Convert 1-d to a 2-d tensor of boxes, which first dim is 1
    if isinstance(boxes, torch.Tensor):
        if len(boxes.shape) == 1:
            boxes = boxes.unsqueeze(0)

        if order == 'xyxy2xywh':
            return torch.cat([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], 1)
        elif order ==  'xywh2xyxy':
            return torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], 1)
        elif order == 'xyxy2cxcy':
            return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2,  # c_x, c_y
                            boxes[:, 2:] - boxes[:, :2]], 1)  # w, h
        elif order == 'cxcy2xyxy':
            return torch.cat([boxes[:, :2] - (boxes[:, 2:] *1.0 / 2),  # x_min, y_min
                            boxes[:, :2] + (boxes[:, 2:] *1.0 / 2)], 1)  # x_max, y_max
    else:
        # Numpy
        new_boxes = boxes.copy()
        if order == 'xywh2xyxy':
            new_boxes[:,2] = boxes[:,0] + boxes[:,2]
            new_boxes[:,3] = boxes[:,1] + boxes[:,3]
            return new_boxes
        elif order == 'xyxy2xywh':
            new_boxes[:,2] = boxes[:,2] - boxes[:,0]
            new_boxes[:,3] = boxes[:,3] - boxes[:,1]
            return new_boxes

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def find_jaccard_overlap(set_1, set_2, order='xyxy'):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    The default box order is (xmin, ymin, xmax, ymax).

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    if order == 'xywh':
        set_1 = change_box_order(set_1, 'xywh2xyxy')
        set_2 = change_box_order(set_2, 'xywh2xyxy')

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)