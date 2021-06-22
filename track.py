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
from time import time 

class VideoTracker():
    def __init__(self, args, config):
        self.video_name = args.video_name
        self.out_path = args.output_path
        self.debug = args.debug
        
        cfg = config.cam[self.video_name]
        cam_cfg = cfg['tracking_config']
        
        # self.zone_path = cfg['zone']
        self.video_path = args.video_path
        self.boxes_path = args.boxes_txt
        self.classes = config.classes
        self.idx_classes = {idx:i for idx,i in enumerate(self.classes)}
        self.num_classes = len(config.classes)

        ## Those polygons and directions are included in the dataset
        # self.polygons, self.directions = self.get_annotations()

        ## Build up a tracker for each class
        self.deepsort = [self.build_tracker(config.checkpoint, cam_cfg) for i in range(self.num_classes)]

        self.out_path = os.path.join(self.out_path, self.video_name)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        output_vid = os.path.join(self.out_path,self.video_name+'_track.mp4')
        self.output_text = os.path.join(self.out_path,self.video_name+'_track.txt')
        self.vidcap = cv2.VideoCapture(self.video_path)

        if self.vidcap.isOpened(): 
            # get vidcap property 
            self.WIDTH  = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
            self.HEIGHT = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
            self.FPS = int(self.vidcap.get(cv2.CAP_PROP_FPS))
            self.NUM_FRAMES = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            print("Video not found")
            return None
        
        if args.viz:
            self.writer = cv2.VideoWriter(
                output_vid,
                cv2.VideoWriter_fourcc(*'mp4v'), 
                self.FPS, 
                (self.WIDTH,self.HEIGHT)
            )

        if args.save_npy:
            self.npy_arr = []
            self.output_npy = os.path.join(self.out_path,self.video_name+'.npy')

        self.load_detection_result()

    def load_detection_result(self):
        self.frame_dets = {}
        with open(self.boxes_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            frame_idx,x1,y1,w,h,label,score = line.split(' ')
            frame_idx = int(frame_idx)
            x1 = int(x1)
            y1 = int(y1)
            w = int(w)
            h = int(h)
            label = int(label)
            score = float(score)
            if frame_idx not in self.frame_dets.keys():
                self.frame_dets[frame_idx] = []
            self.frame_dets[frame_idx].append([x1,y1,w,h,label,score])

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

    def write_to_txt(self, frame_id, outputs, label):
        # output: x1,y1,x2,y2,track_id, track_feat, score
        with open(self.output_text, 'a+') as result_file:
            for obj in outputs:
                track_id = obj[4]
                x1,y1,x2,y2 = obj[:4]
                result_file.write(f'{frame_id} {track_id} {label} {x1} {y1} {x2} {y2}\n')
                # frame_id, obj_id, cls_id, xmin, ymin, xmax, ymax 

    def write_to_npy(self, frame_id, outputs, label):
        #image_id, track_id, bb_left, bb_top, bb_width, bb_height, score, category, feature
        for obj in outputs:
            track_id = obj[4]
            x1,y1,x2,y2 = obj[:4]
            w = x2-x1
            h = y2-y1
            track_feat = obj[5]
            score = obj[6]
            self.npy_arr.append([frame_id, track_id, x1,y1,w,h,score,label,track_feat])
        

    def run(self):
        # Dict to save object's tracks per class
        self.obj_track = [{} for i in range(self.num_classes)]
        idx_frame = 0
    
        with tqdm(total=self.NUM_FRAMES ) as pbar:
            while self.vidcap.isOpened():
                success, im = self.vidcap.read()
                if idx_frame in self.frame_dets.keys():
                    annos = self.frame_dets[idx_frame] # x1,y1,w,h,label,score
                else:
                    annos = None
                if not success:
                    break

                ## Draw polygons to frame
                ori_img = im[..., ::-1]

                im_moi = im.copy()
                cv2.putText(im_moi,str(idx_frame), (10,30), cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,0) , 2)

                if annos is not None:
                    bbox_xyxys = []
                    cls_confs = []
                    cls_ids = []

                    for x1,y1,w,h,label,score in annos:
                        x2 = x1+w
                        y2 = y1+h
                        bbox_xyxys.append([x1,y1,x2,y2])
                        cls_confs.append(score)
                        cls_ids.append(label)
                    bbox_xyxy_ = np.array(bbox_xyxys)
                    cls_conf_ = np.array(cls_confs)
                    cls_id_ = np.array(cls_ids)

                    # AIC class index starts from 1
                    cls_id_ = cls_id_ - 1

                    for i in range(self.num_classes):
                        
                        mask = (cls_id_ == i)
                        
                        bbox_xyxy = bbox_xyxy_[mask]
                        cls_conf = cls_conf_[mask]
                        cls_id = cls_id_[mask]

                        if len(cls_id) > 0:
                            
                            outputs = self.deepsort[i].update(bbox_xyxy, cls_conf, ori_img)

                            self.write_to_txt(idx_frame, outputs, i)

                            if args.viz:
                                im_show = draw_re_id(outputs, im_moi, labels=i)
                        else:
                            if args.viz:
                                im_show  = im_moi
                else:
                    if args.viz:
                        im_show = im_moi
                    
                if args.viz:
                    self.writer.write(im_show)
                
                idx_frame += 1
                pbar.update(1)

        if args.save_npy:
            np.save(self.output_npy, self.npy_arr, allow_pickle=True)

    

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
    parser.add_argument('video_name',  help='configuration cam file')
    parser.add_argument('--output_path', help='output path')
    parser.add_argument('--video_path', type=str, help='output path')
    parser.add_argument('--boxes_txt', type=str, help='output path')
    parser.add_argument('--viz', action='store_true', default=False, help='output path')
    parser.add_argument('--debug', action='store_true', default = False,help='debug print object id to file')          
    parser.add_argument('--save_npy', action='store_true', default = False,help='debug print object id to file')          

    args = parser.parse_args()
    configs = Config('configs/cam_configs.yaml')
    tracker = VideoTracker(args, configs)
    tracker.run()
    running_time = time.time() - start
    with open('time.txt', 'a+') as f:
        f.write(f"{running_time} ")

