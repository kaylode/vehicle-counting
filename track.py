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

class VideoTracker():
    def __init__(self, args, config):
        self.video_name = args.video_name #cam_01
        self.out_path = args.out_path
        self.cam_id = int(self.video_name[-2:])
        self.display = args.display
        
        cfg = config.cam[self.video_name]
        cam_cfg = cfg['tracking_config']
        
       
        self.frame_start = args.frame_start
        self.frame_end = args.frame_end
        self.zone_path = cfg['zone']
        self.video_path = cfg['video']
        self.boxes_path = cfg['boxes']
        self.classes = config.classes
        self.idx_classes = {idx:i for idx,i in enumerate(self.classes)}
        self.num_classes = len(config.classes)
        self.width, self.height = config.size
        self.polygons, self.directions = self.get_annotations()
        self.deepsort = [self.build_tracker(config.checkpoint, cam_cfg) for i in range(self.num_classes)]


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

    def submit(self, video_name, moi_detections):
        submission_path = os.path.join(self.out_path, 'submission')
        if not os.path.exists(submission_path):
            os.mkdir(submission_path)
        file_name = os.path.join(submission_path, self.video_name)
        result_filename = '{}.txt'.format(file_name)
        result_debug = '{}_debug.txt'.format(file_name)
        with open(result_filename, 'w+') as result_file, open(result_debug, 'w+') as debug_file:
            for obj_id , frame_id, movement_id, vehicle_class_id in moi_detections:
                result_file.write('{} {} {} {}\n'.format(video_name, frame_id, str(int(movement_id)), vehicle_class_id+1))
                if self.display:
                    debug_file.write('{} {} {} {} {}\n'.format(obj_id, video_name, frame_id, str(int(movement_id)), vehicle_class_id+1))
        print('Save to',result_filename,'and', result_debug)

    def run(self):
        self.obj_track = [{}, {}, {}, {}]
        vidcap = cv2.VideoCapture(self.video_path)
        idx_frame = 0
        try:
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
                    bbox_xyxy_ = np.array(bbox_xyxy[mask==1])
                    cls_conf_ = np.array(cls_conf[mask==1])
                    cls_ids_ = np.array(cls_ids[mask==1])

                    for i in range(self.num_classes):
                        
                        mask = cls_ids_ == i
                        bbox_xyxy = bbox_xyxy_[mask]
                        cls_conf = cls_conf_[mask]
                        cls_ids = cls_ids_[mask]

                        if len(cls_ids) > 0:
                            outputs = self.deepsort[i].update(bbox_xyxy, cls_conf, ori_img)
                            outputs = self.deepsort[i].update(bbox_xyxy, cls_conf, ori_img)
                            for obj in outputs:
                                identity = obj[-1]
                                center = [(obj[2]+obj[0]) / 2, (obj[3] + obj[1])/2]
                                label = obj[-2]
                                if identity not in self.obj_track[i]:
                                    self.obj_track[i][identity] = {
                                        'labels': [label],
                                        'coords': [center],
                                        'frame_id': [idx_frame]
                                    }
                                else:
                                    self.obj_track[i][identity]['labels'].append(label)
                                    self.obj_track[i][identity]['coords'].append(center)
                                    self.obj_track[i][identity]['frame_id'].append(idx_frame)
                    
                            im_show = re_id(outputs, im_moi, labels=i)
                        else:
                            im_show = im_moi

                    if self.display:
                        self.writer.write(im_show)
                    
                    idx_frame += 1
                    pbar.update(1)
        except KeyboardInterrupt:
            pass
        # obj id, last frame id, movement id, vehicle id
        
        
        moi_detections = counting_moi(self.directions ,self.obj_track, self.polygons, self.cam_id)
        self.submit(self.video_name, moi_detections)


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
    parser.add_argument('--display', action='store_true', default = False,
                        help='debug print object id to file')          
    args = parser.parse_args()
    configs = Config(args.config)
    tracker = VideoTracker(args, configs)
    tracker.run()