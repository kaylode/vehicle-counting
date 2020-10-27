import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import os
from utils.utils import *
from PIL import Image
import matplotlib.pyplot as plt
from models.deepsort.deep_sort import DeepSort
from tqdm import tqdm
import argparse
import json
import cv2
from configs import Config

frame_width = 1280
frame_height = 720




vehicle_id = {
        1: 'car',
        0: 'motorcycle',
        2: 'bus',
        3: 'truck'
    }


def main(args, config):
    path = args.box_path
    ann_path= args.ann_path
    video_path = args.video_path
    debug_mode = args.debug

    video_name = path[-6:]
    

    cfg = config.cam[video_name]
    
    deepsort = DeepSort(
        cfg['REID_CKPT'], 
        max_dist=cfg['MAX_DIST'],
        min_confidence=cfg['MIN_CONFIDENCE'], 
        nms_max_overlap=cfg['NMS_MAX_OVERLAP'],
        max_iou_distance=cfg['MAX_IOU_DISTANCE'], 
        max_age=cfg['MAX_AGE'],
        n_init=cfg['N_INIT'],
        nn_budget=cfg['NN_BUDGET'],
        use_cuda=1)


    if not os.path.exists('results/videos'):
        os.mkdir('results/videos')
    output_vid = os.path.join('results/videos/',video_name+'.mp4')
    outvid = cv2.VideoWriter(output_vid,cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))

    root =  ann_path
    
    
    cam_id = int(video_name[-2:])
    direct = get_directions(root, video_name)
    polygons = get_zone(root, video_name)
    
 
    num_frame = len(os.listdir(path))
    #frame_names = [str(i).zfill(5)+'.json' for i in range(5892)]
    #idx_frame = int(frame_names[0][:-5])
    #det_paths = [os.path.join(path, i) for i in frame_names]
   
    
    obj_track = {}
    vidcap = cv2.VideoCapture(video_path)
    idx_frame = 0
    with tqdm(total=5892) as pbar:
        while vidcap.isOpened():
            success, im = vidcap.read()
            anno = os.path.join(path, str(idx_frame).zfill(5) + '.json')
            if not success:
                break
            ori_img = im[..., ::-1]
            overlay_moi = im.copy()
            alpha = 0.2
            cv2.fillConvexPoly(overlay_moi, np.array(polygons).astype(int), (255,255,0))
            im_moi = cv2.addWeighted(overlay_moi, alpha, im, 1 - alpha, 0)
            cv2.putText(im_moi,str(idx_frame), (10,30), cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,0) , 2)

            try:
                with open(anno, 'r') as f:
                    objs = json.load(f)
            except FileNotFoundError:
                print("Tracked {} frames".format(num_frame))
                break
            
            bbox_xyxy = np.array(objs['bboxes'])
            cls_conf = np.array(objs['scores'])
            cls_ids = np.array(objs['classes'])

            # Check only bbox in roi
            mask = np.array([1 if check_bbox_intersect_polygon(polygons,i.tolist()) else 0 for i in bbox_xyxy])
            bbox_xyxy = np.array(bbox_xyxy[mask==1])
            cls_conf = np.array(cls_conf[mask==1])
            cls_ids = np.array(cls_ids[mask==1])

            if len(cls_ids) > 0:
                outputs = deepsort.update(bbox_xyxy, cls_conf, ori_img, cls_ids)
                outputs = deepsort.update(bbox_xyxy, cls_conf, ori_img, cls_ids)
                for obj in outputs:
                    identity = obj[-1]
                    center = [(obj[2]+obj[0]) / 2, (obj[3] + obj[1])/2]
                    label = obj[-2]
                    if identity not in obj_track:
                        obj_track[identity] = {
                            'labels': [label],
                            'coords': [center],
                            'frame_id': [idx_frame]
                        }
                    else:
                        obj_track[identity]['labels'].append(label)
                        obj_track[identity]['coords'].append(center)
                        obj_track[identity]['frame_id'].append(idx_frame)
        
                im_show = re_id(outputs, im_moi, vehicle_id)
            else:
                im_show = im_moi

            if output_vid is not None:
                outvid.write(im_show)
            else:
                cv2.imshow("tracking", im_show)
                cv2.waitKey(1)
            idx_frame += 1
            pbar.update(1)
    # obj id, last frame id, movement id, vehicle id
    
    
    moi_detections = counting_moi(direct,obj_track, polygons, cam_id)
    submit(video_name, output_vid,moi_detections, debug = debug_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
    parser.add_argument('--box_path', type=str, 
                        help='path to detection folder')
    parser.add_argument('--ann_path', type=str, 
                        help='path to zone folder')
    parser.add_argument('--video_path', type=str, 
                        help='path to image folder')
    parser.add_argument('--output_path', type=str, default = 'results',
                        help='name of output results')
    parser.add_argument('--debug', action='store_true', default = False,
                        help='debug print object id to file')
    parser.add_argument('--config', default = './configs/cam_configs.yaml',
                        help='configuration cam file')

    args = parser.parse_args()
    configs = Config(args.config)

    main(args, configs)


    
                
