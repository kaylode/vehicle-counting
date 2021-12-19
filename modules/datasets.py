import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentations.transforms import get_resize_augmentation, MEAN, STD

from utilities.utils import write_to_video
from utilities.counting import visualize_merged

class VideoSet:
    def __init__(self, config, input_path):
        self.input_path = input_path # path to video file
        self.image_size = config.image_size
        self.transforms = A.Compose([
            get_resize_augmentation(config.image_size, keep_ratio=config.keep_ratio),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
        ])

        self.initialize_stream()

    def initialize_stream(self):
        self.stream = cv2.VideoCapture(self.input_path)
        self.current_frame_id = 0
        self.video_info = {}

        if self.stream.isOpened(): 
            # get self.stream property 
            self.WIDTH  = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
            self.HEIGHT = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
            self.FPS = int(self.stream.get(cv2.CAP_PROP_FPS))
            self.NUM_FRAMES = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_info = {
                'name': os.path.basename(self.input_path),
                'width': self.WIDTH,
                'height': self.HEIGHT,
                'fps': self.FPS,
                'num_frames': self.NUM_FRAMES
            }
        else:
            assert 0, f"Cannot read video {os.path.basename(self.input_path)}"

    def __getitem__(self, idx):
        success, ori_frame = self.stream.read()
        if not success:
            print(f"Cannot read frame {self.current_frame_id} from {self.video_info['name']}")
            self.current_frame_id = idx+1
            return None
        
        self.current_frame_id = idx+1
        frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB) 

        return {
            'img': frame,
            'frame': self.current_frame_id,
            'ori_img': ori_frame,
        }

    def collate_fn(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        imgs = [s['img'] for s in batch]
        ori_imgs = [s['ori_img'] for s in batch]
        frames = [s['frame'] for s in batch]

        return {
            'imgs': imgs,
            'frames': frames,
            'ori_imgs': ori_imgs,
        }

    def __len__(self):
        return self.NUM_FRAMES

    def __str__(self):
        s2 = f"Number of frames: {self.NUM_FRAMES}"
        return s2

class VideoLoader(DataLoader):
    def __init__(self, config, video_path):
        self.video_path = video_path
        dataset = VideoSet(config, video_path)
        self.video_info = dataset.video_info
       
        super(VideoLoader, self).__init__(
            dataset,
            batch_size= 1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            collate_fn= dataset.collate_fn)

    def reinitialize_stream(self):
        self.dataset.initialize_stream()
        
class VideoWriter:
    def __init__(self, video_info, saved_path, obj_list):
        self.video_info = video_info
        self.saved_path = saved_path
        self.obj_list = obj_list

        if not os.path.exists(self.saved_path):
            os.makedirs(self.saved_path, exist_ok=True)
            
        video_name = self.video_info['name']
        outpath =os.path.join(self.saved_path, video_name)
        self.FPS = self.video_info['fps']
        self.WIDTH = self.video_info['width']
        self.HEIGHT = self.video_info['height']
        self.NUM_FRAMES = self.video_info['num_frames']
        self.outvid = cv2.VideoWriter(
            outpath,   
            cv2.VideoWriter_fourcc(*'mp4v'), 
            self.FPS, 
            (self.WIDTH, self.HEIGHT))

    def write(self, img, boxes, labels, scores=None, tracks=None):
        write_to_video(
            img, boxes, labels, 
            scores = scores,
            tracks=tracks, 
            imshow=False, 
            outvid = self.outvid, 
            obj_list=self.obj_list)

    def write_full_to_video(
        self,
        videoloader,
        num_classes,
        csv_path,
        paths, polygons):

        visualize_merged(
            videoloader, 
            csv_path, 
            num_classes=num_classes,
            directions = paths, 
            zones = polygons, 
            outvid=self.outvid
        )