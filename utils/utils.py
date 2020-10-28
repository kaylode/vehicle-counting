import math
import os
import uuid
from glob import glob
from typing import Union
import json
import cv2
import numpy as np
import torch
import webcolors
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
from torchvision.ops.boxes import batched_nms
from .bb_polygon import is_bounding_box_intersect
import torchvision
import sys
sys.path.append('..')


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


def postprocessing(outs, imgs, retransforms = None):
    for item in outs:
        
        boxes_out = item['bboxes']
        boxes_out_xywh = change_box_order(boxes_out, order = 'xyxy2xywh')
        new_boxes = retransforms(img = imgs, box=boxes_out_xywh)['box']
        new_boxes = change_box_order(new_boxes, order = 'xywh2xyxy')
        item['bboxes'] = new_boxes

    return outs
        

def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])

def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            if "conv_list" or "header" in name:
                variance_scaling_(module.weight.data)
            else:
                nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    torch.nn.init.constant_(module.bias, bias_value)
                else:
                    module.bias.data.zero_()


def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)


STANDARD_COLORS = [
    'LawnGreen', 'LightBlue' , 'Crimson', 'Gold', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)


color_list = standard_to_bgr(STANDARD_COLORS)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

# Additionals

def re_id(outputs, ori_img, labels, polygons=None):
    results= []
    if len(outputs) > 0:
        bbox_tlwh = []
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        ori_im = draw_boxes(ori_img, bbox_xyxy, identities, labels)

    else:
        ori_im = ori_img
    return ori_im


def draw_boxes(img, bbox, identities=None, labels = None):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0  
        color = color_list[labels]
        label = '{}id:{:d} c:{}'.format("", id, str(labels+1))
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        try:
            cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
            cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
            cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
        except:
            pass
    return img


def display_img(preds, imgs, imshow=True,  outvid = None, obj_list=None):
    
    for i in range(len(imgs)):
        if len(preds[i]['bboxes']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['bboxes'])):
            x1, y1, x2, y2 = preds[i]['bboxes'][j].astype(np.int)
            obj = obj_list[preds[i]['classes'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[preds[i]['classes'][j]])


        if imshow:
          
            cv2.namedWindow('img',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 600,600)
            cv2.imshow('img', imgs[i])
            cv2.waitKey(1)


        if outvid is not None:
            outvid.write(imgs[i])


def cosin_similarity(a2d, b2d):
    a = np.array((a2d[1][0] - a2d[0][0], a2d[1][1 ]- a2d[0][1]))
    b = np.array((b2d[1][0] - b2d[0][1], b2d[1][1] - b2d[1][0]))
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))



def distance_point_line(line, p3): # line(p1,p2), point(p3)
    p1, p2 = line
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)


def counting_moi_cosine(paths, obj_tracks, polygons):
    moi_detection_list = []
    for i in range(len(obj_tracks)):
        for obj_id, obj_list in obj_tracks[i].items():
            
            # Most frequency class
            true_label = max(set(obj_list['labels']), key = obj_list['labels'].count) 
            last_frame = obj_list['frame_id'][-1]

            max_cosin = -2
            movement_id = 1
            first_point = obj_list['coords'][0]
            last_point = obj_list['coords'][-1]
            obj_vector = (first_point, last_point)
            for movement_label, movement_vector in paths.items():
                cosin = cosin_similarity(obj_vector,movement_vector)
                if cosin > max_cosin:
                    max_cosin = cosin
                    movement_id = movement_label

            moi_detection_list.append((obj_id, last_frame, movement_id, true_label))    
    return moi_detection_list


def counting_moi_distance(paths, obj_tracks, polygons, cam_id):
    """
    Args:
    paths: List of MOI - (first_point, last_point)
    moto_vector_list: List of tuples (first_point, last_point, last_frame_id) 

    Returns:
    A list of tuples (frame_id, movement_id, vehicle_class_id)
    """

    s = [[polygons[i],polygons[0]] if i == len(polygons)-1 else [polygons[i], polygons[i+1]] for i in range(len(polygons))]
    movement_dict = {
        7: [[None,None,'06','01'], [None,None,'06','01'], [None,'03',None,'04'], [None,'02','05', None]],
        9: [['05','01','01','04'], ['02',None,'01','03'], ['02','02','06','03'], ['05','06','06', '04']],
        10: [['03','04',None,'05'], ['03','02','01','01'], ['06','02',None,'01'], ['06','02','02', '05']],
        
    }
    movements = movement_dict[cam_id]
    moi_detection_list = []
    for i in range(len(obj_tracks)):
        for obj_id, obj_list in obj_tracks[i].items():
            
            # Most frequency class
            true_label = max(set(obj_list['labels']), key = obj_list['labels'].count) 
            last_frame = obj_list['frame_id'][-1]

            movement_id = 1
            first_point = obj_list['coords'][0]
            last_point = obj_list['coords'][-1]

            minn = 10000
            minn2 = 10000
            min_id = None
            min_id2 = None
            for idx, i in enumerate(s):
                dist1 = distance_point_line(i, first_point)
                dist2 = distance_point_line(i, last_point)
                if  dist1 <= minn:
                    minn = dist1
                    min_id = idx
                if  dist2 <= minn2:
                    minn2 = dist2
                    min_id2 = idx

            if min_id is None or min_id2 is None:
                continue
            movement_id = movements[min_id][min_id2]
            if movement_id is None:
                continue

            moi_detection_list.append((obj_id, last_frame, movement_id, true_label))
    return moi_detection_list


def counting_moi(paths, obj_tracks, polygons, cam_id):
    intersect = [7,9,10]
    if cam_id in intersect:
        moi_detection_list = counting_moi_distance(paths, obj_tracks, polygons, cam_id)
    else:
        moi_detection_list = counting_moi_cosine(paths, obj_tracks, polygons)
    return moi_detection_list


def check_bbox_intersect_polygon(polygon, bbox):
    """
    
    Args:
        polygon: List of points (x,y)
        bbox: A tuple (xmin, ymin, xmax, ymax)
    
    Returns:
        True if the bbox intersect the polygon
    """
    x1, y1, x2, y2 = bbox
    bb = [(x1,y1), (x2, y1), (x2,y2), (x1,y2)]
    return is_bounding_box_intersect(bb, polygon)
