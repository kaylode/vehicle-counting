import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import webcolors
import cv2
import gdown
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_

STANDARD_COLORS = [
    'LawnGreen', 'LightBlue' , 'Crimson', 'Gold', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite','Azure', 
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

color_list = standard_to_bgr(STANDARD_COLORS)

def draw_boxes_v2(img_name, img, boxes, labels, scores, obj_list=None, figsize=(15,15)):
    """
    Visualize an image with its bouding boxes
    """
    fig,ax = plt.subplots(figsize=figsize)

    if isinstance(img, torch.Tensor):
        img = img.numpy().squeeze().transpose((1,2,0))
    # Display the image
    ax.imshow(img)

    # Create a Rectangle patch
    for box, label, score in zip(boxes, labels, scores):
        label = int(label)
        color = STANDARD_COLORS[label]
        x,y,w,h = box
        rect = patches.Rectangle((x,y),w,h,linewidth=1.5,edgecolor = color,facecolor='none')
        score = np.round(score, 3)
        if obj_list is not None:
            text = '{}: {}'.format(obj_list[label], str(score))
        else:
            text = '{}: {}'.format(label, str(score))
        plt.text(x, y-3,text, color = color, fontsize=15)
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.axis('off')
    plt.savefig(img_name,bbox_inches='tight')
    plt.close()

def draw_pred_gt_boxes(image_outname, img, boxes, labels, scores, image_name=None, figsize=(10,10)):
    """
    Visualize an image with its bouding boxes
    """
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    # if image_name is not None:
    #     fig.suptitle(image_name)
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if isinstance(img, torch.Tensor):
        img = img.numpy().squeeze().transpose((1,2,0))
    # Display the image
    ax1.imshow(img)
    ax2.imshow(img)
    
    ax1.set_title('Prediction')
    ax2.set_title('Ground Truth')

    # Split prediction  and ground truth
    pred_boxes, pred_labels, pred_scores = boxes[0], labels[0], scores
    gt_boxes, gt_labels = boxes[1], labels[1]

    # Plot prediction boxes
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        label = int(label)
        color = STANDARD_COLORS[label]
        x,y,w,h = box
        rect = patches.Rectangle((x,y),w,h,linewidth=1.5,edgecolor = color,facecolor='none')
        score = np.round(score, 3)
        text = '{}: {}'.format(label, str(score))
        ax1.text(x, y-3,text, color = color, fontsize=15)
        # Add the patch to the Axes
        ax1.add_patch(rect)

    # Plot ground truth boxes
    for box, label in zip(gt_boxes, gt_labels):
        label = int(label)
        if label <0:
            continue
        color = STANDARD_COLORS[label]
        x,y,w,h = box
        rect = patches.Rectangle((x,y),w,h,linewidth=1.5,edgecolor = color,facecolor='none')
        score = np.round(score, 3)
        text = '{}'.format(label)
        ax2.text(x, y-3,text, color = color, fontsize=15)
        # Add the patch to the Axes
        ax2.add_patch(rect)

    plt.axis('off')
    fig = ax1.get_figure()
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    fig = ax2.get_figure()
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    plt.savefig(image_outname,bbox_inches='tight')
    return fig

    # plt.close()

def write_to_video(img, boxes, labels, scores=None, tracks=None, imshow=True,  outvid = None, obj_list=None):
    
    def plot_one_box(img, box, key=None, value=None, color=None, line_thickness=None):
        tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness

        coord = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
        c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl*2)
        if key is not None and value is not None:
            header = f'{key} || {value}'
            tf = max(tl - 2, 1)  # font thickness
            s_size = cv2.getTextSize(f'| {value}', 0, fontScale=float(tl) / 3, thickness=tf)[0]
            t_size = cv2.getTextSize(f'{key} |', 0, fontScale=float(tl) / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, header, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                        thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

    boxes = boxes.astype(np.int)

    if scores is not None:
        for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            plot_one_box(
                img, 
                box, 
                key = obj_list[int(label)],
                value =  '{:.0%}'.format(float(score)),
                color=color_list[int(label)])
    if tracks is not None:
        for idx, (box, track, label) in enumerate(zip(boxes, tracks, labels)):
            plot_one_box(
                img, 
                box, 
                key=f'id: {track}',
                value=f'cls: {label}',
                color=color_list[int(label)])

    if imshow:
        cv2.namedWindow('img',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600,600)
        cv2.imshow('img', img)
        cv2.waitKey(1)

    if outvid is not None:
        outvid.write(img)

def download_drive(id_or_url, cached=None, md5=None, quiet=False):
    if id_or_url.startswith('http'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?id={}'.format(id_or_url)

    return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)

def download_hub(url, dest):
    torch.hub.download_url_to_file(url, str(dest))
    return dest

def download_weights(url, dest):
    return download_hub(url, dest)

weight_url = {
    "yolov5s": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt" ,
    "yolov5m": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt" ,
    "yolov5l": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l.pt",
    "yolov5x": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x.pt",
}

def download_pretrained_weights(name, cached=None):
    return download_weights(weight_url[name], cached)
    
