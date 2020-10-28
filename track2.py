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

class VideoTracker():
    def __init__(self, args, config):
        self.video_name = args.video_name[:-4] #cam_01.mp4
        self.out_path = args.out_path
        self.cam_id = int(self.video_name[-2:])
        self.display = args.display
        
        cfg = config.cam[self.video_name]
        cam_cfg = cfg['tracking_config']
        
        self.debug_mode = args.debug
        self.frame_start = args.frame_start
        self.frame_end = args.frame_end
        self.zone_path = cfg['zone']
        self.video_path = cfg['video']
        self.boxes_path = cfg['boxes']
        self.num_classes = config.num_classes
        self.width, self.height = config.size
        self.polygons, self.directions = self.get_annotations()
        self.deepsort = [self.build_tracker(cfg.checkpoint, cam_cfg) for i in range(self.num_classes)]


        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        output_vid = os.path.join(self.out_path,self.video_name+'.mp4')
        self.writer = cv2.VideoWriter(output_vid,cv2.VideoWriter_fourcc(*'mp4v'), 10, (self.width,self.height))
        
    def build_tracker(self, checkpoint, cam_cfg):
        return DeepSort(
                checkpoint, 
                max_dist=cam_cfg['MAX_DIST'],
                min_confidence=cam_cfg['MIN_CONFIDENCE'], 
                nms_max_overlap=cam_cfg['NMS_MAX_OVERLAP'],
                max_iou_distance=cam_cfg['MAX_IOU_DISTANCE'], 
                max_age=cam_cfg['MAX_AGE'],
                n_init=cam_cfg['N_INIT'],
                nn_budget=cam_cfg['NN_BUDGET'],
                use_cuda=1)

    def get_annotations(self):
        with open(self.zone_path, 'r') as f:
            anno = json.load(f)
        
        directions =  {}
        zone = anno['shapes'][0]['points']
        for i in anno['shapes']:
            if i['label'].startswith('direction'):
                directions[i['label'][-2:]] = i['points']
        return zone, directions

    def run(self):
        obj_track = {}
        vidcap = cv2.VideoCapture(self.video_path)
        idx_frame = 0
        with tqdm(total=self.frame_end) as pbar:
            while vidcap.isOpened():
                success, im = vidcap.read()
                if idx_frame < self.frame_start:
                    idx_frame+=1
                    continue
                anno = os.path.join(self.boxes_path, str(idx_frame).zfill(5) + '.json')
                if not success:
                    break
                ori_img = im[..., ::-1]
                overlay_moi = im.copy()
                alpha = 0.2
                cv2.fillConvexPoly(overlay_moi, np.array(self.polygons).astype(int), (255,255,0))
                im_moi = cv2.addWeighted(overlay_moi, alpha, im, 1 - alpha, 0)
                cv2.putText(im_moi,str(idx_frame), (10,30), cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,0) , 2)

                try:
                    with open(anno, 'r') as f:
                        objs = json.load(f)
                except FileNotFoundError:
                    print("Tracked {} frames".format(idx_frame+1))
                    break
                
                bbox_xyxy = np.array(objs['bboxes'])
                cls_conf = np.array(objs['scores'])
                cls_ids = np.array(objs['classes'])

                # Check only bbox in roi
                mask = np.array([1 if check_bbox_intersect_polygon(self.polygons,i.tolist()) else 0 for i in bbox_xyxy])
                bbox_xyxy = np.array(bbox_xyxy[mask==1])
                cls_conf = np.array(cls_conf[mask==1])
                cls_ids = np.array(cls_ids[mask==1])

                for i in range(len(self.num_classes)):
                    if len(cls_ids) > 0:
                        outputs = self.deepsort[i].update(bbox_xyxy, cls_conf, ori_img, cls_ids)
                        outputs = self.deepsort[i].update(bbox_xyxy, cls_conf, ori_img, cls_ids)
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

                if self.display:
                  self.writer.write(im_show)
                
                idx_frame += 1
                pbar.update(1)
        # obj id, last frame id, movement id, vehicle id
        
        
        moi_detections = counting_moi(self.directions ,obj_track, self.polygons, self.cam_id)
        submit(self.video_name, output_vid,moi_detections, debug = self.debug_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
    parser.add_argument('video_name',
                        help='configuration cam file')
    parser.add_argument('--out_path',
                        help='output path') 
    parser.add_argument('--frame_start', default = 0,
                        help='start at frame')
    parser.add_argument('--frame_end', default = 5892,
                        help='end at frame')
    parser.add_argument('--config', default = './configs/cam_configs.yaml',
                        help='configuration cam file')
    parser.add_argument('--debug', action='store_true', default = False,
                        help='debug print object id to file')
    parser.add_argument('--display', action='store_true', default = False,
                        help='debug print object id to file')          
    args = parser.parse_args()
    configs = Config(args.config)
    tracker = VideoTracker(args, configs)
    tracker.run()