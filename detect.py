import random
from utils.getter import *
import argparse
import os
import cv2
from tqdm import tqdm

from augmentations.transforms import MEAN, STD
from models.deepsort.deep_sort import DeepSort
from utils.counting import (
    check_bbox_intersect_polygon, 
    save_tracking_to_csv, load_zone_anno,
    find_best_match_direction, visualize_merged)

parser = argparse.ArgumentParser(description='Perfom Objet Detection')
parser.add_argument('--weight', type=str, default = None,help='version of EfficentDet')
parser.add_argument('--input_path', type=str, help='path to an image to inference')
parser.add_argument('--output_path', type=str, help='path to save inferenced image')
parser.add_argument('--gpus', type=str, default='0', help='path to save inferenced image')
parser.add_argument('--min_conf', type=float, default= 0.1, help='minimum confidence for an object to be detect')
parser.add_argument('--min_iou', type=float, default=0.5, help='minimum iou threshold for non max suppression')
parser.add_argument('--tta', action='store_true', help='whether to use test time augmentation')
parser.add_argument('--tta_ensemble_mode', type=str, default='wbf', help='tta ensemble mode')
parser.add_argument('--tta_conf_threshold', type=float, default=0.01, help='tta confidence score threshold')
parser.add_argument('--tta_iou_threshold', type=float, default=0.9, help='tta iou threshold')
parser.add_argument('--debug', action='store_true', help='save detection at')


class VideoSet:
    def __init__(self, config, input_path):
        self.input_path = input_path # path to video file
        self.image_size = config.image_size
        self.transforms = A.Compose([
            get_resize_augmentation(config.image_size, keep_ratio=config.keep_ratio),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
            ToTensorV2(p=1.0)
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
            raise f"Cannot read video {os.path.basename(self.input_path)}"

    def __getitem__(self, idx):
        success, ori_frame = self.stream.read()
        if not success:
            print(f"Cannot read frame {self.current_frame_id} from {self.video_info['name']}")
            return None
        else:
            self.current_frame_id = idx+1
        frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame /= 255.0
        if self.transforms is not None:
            inputs = self.transforms(image=frame)['image']

        image_w, image_h = self.image_size
        ori_height, ori_width, _ = ori_frame.shape

        return {
            'img': inputs,
            'frame': self.current_frame_id,
            'ori_img': ori_frame,
            'image_ori_w': ori_width,
            'image_ori_h': ori_height,
            'image_w': image_w,
            'image_h': image_h,
        }

    def collate_fn(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        imgs = torch.stack([s['img'] for s in batch])   
        ori_imgs = [s['ori_img'] for s in batch]
        frames = [s['frame'] for s in batch]
        image_ori_ws = [s['image_ori_w'] for s in batch]
        image_ori_hs = [s['image_ori_h'] for s in batch]
        image_ws = [s['image_w'] for s in batch]
        image_hs = [s['image_h'] for s in batch]
        img_scales = torch.tensor([1.0]*len(batch), dtype=torch.float)
        img_sizes = torch.tensor([imgs[0].shape[-2:]]*len(batch), dtype=torch.float)   

        return {
            'imgs': imgs,
            'frames': frames,
            'ori_imgs': ori_imgs,
            'image_ori_ws': image_ori_ws,
            'image_ori_hs': image_ori_hs,
            'image_ws': image_ws,
            'image_hs': image_hs,
            'img_sizes': img_sizes, 
            'img_scales': img_scales
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
        csv_path,
        paths, polygons):

        visualize_merged(
            videoloader, 
            csv_path, 
            directions = paths, 
            zones = polygons, 
            outvid=self.outvid
        )
            

class VideoDetect:
    def __init__(self, args, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')   
        self.debug = args.debug
        self.config = config
        self.min_iou = args.min_iou
        self.min_conf = args.min_conf
        self.max_dets=config.max_post_nms
        self.keep_ratio=config.keep_ratio
        self.fusion_mode=config.fusion_mode

        if args.tta:
            self.tta = TTA(
                min_conf=args.tta_conf_threshold, 
                min_iou=args.tta_iou_threshold, 
                postprocess_mode=args.tta_ensemble_mode)
        else:
            self.tta = None

        if args.weight is not None:
            self.class_names, num_classes = get_class_names(args.weight)
        self.class_names.insert(0, 'Background')

        net = get_model(
            args, config,
            num_classes=num_classes)

        self.num_classes = num_classes
        self.model = Detector(model = net, device = self.device)
        self.model.eval()

        if args.weight is not None:                
            load_checkpoint(self.model, args.weight)

    def run(self, batch):
        with torch.no_grad():
            boxes_result = []
            labels_result = []
            scores_result = []
                
            if self.tta is not None:
                preds = self.tta.make_tta_predictions(self.model, batch)
            else:
                preds = self.model.inference_step(batch)

            for idx, outputs in enumerate(preds):
                img_w = batch['image_ws'][idx]
                img_h = batch['image_hs'][idx]
                img_ori_ws = batch['image_ori_ws'][idx]
                img_ori_hs = batch['image_ori_hs'][idx]
                
                outputs = postprocessing(
                    outputs, 
                    current_img_size=[img_w, img_h],
                    ori_img_size=[img_ori_ws, img_ori_hs],
                    min_iou=self.min_iou,
                    min_conf=self.min_conf,
                    max_dets=self.max_dets,
                    keep_ratio=self.keep_ratio,
                    output_format='xywh',
                    mode=self.fusion_mode)

                boxes = outputs['bboxes'] 

                # Here, labels start from 1, but will subtract 1 later
                labels = outputs['classes'] 
                scores = outputs['scores']

                boxes_result.append(boxes)
                labels_result.append(labels)
                scores_result.append(scores)

        return {
            "boxes": boxes_result, 
            "labels": labels_result,
            "scores": scores_result }

class VideoTracker:
    def __init__(self, num_classes, cam_config, video_info, deepsort_chepoint):
        tracking_config = cam_config["tracking_config"]
        self.num_classes = num_classes 
        self.video_info = video_info
        self.num_frames = video_info['num_frames']

        ## Build up a tracker for each class
        self.deepsort = [self.build_tracker(deepsort_chepoint, tracking_config) for i in range(num_classes)]

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

    def run(self, image, boxes, labels, scores):
        # Dict to save object's tracks per class
        # boxes: xywh
        self.obj_track = [{} for i in range(self.num_classes)]

        ## Draw polygons to frame
         
        # cv2.putText(im_moi,str(frame_id), (10,30), cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,0) , 2)

        bbox_xyxy = boxes.copy()
        bbox_xyxy[:, 2] += bbox_xyxy[:, 0]
        bbox_xyxy[:, 3] += bbox_xyxy[:, 1]

        result_dict = {
            'tracks': [],
            'boxes': [],
            'labels': [],
            'scores': []
        }
        labels__ = labels.copy() - 1
        for i in range(self.num_classes):
            mask = (labels__ == i)     
            bbox_xyxy_ = bbox_xyxy[mask]
            scores_ = scores[mask]
            labels_ = labels__[mask]

            if len(labels_) > 0:

                # output: x1,y1,x2,y2,track_id, track_feat, score
                outputs = self.deepsort[i].update(bbox_xyxy_, scores_, image)
                
                for obj in outputs:
                    box = obj[:4]
                    result_dict['tracks'].append(obj[4])
                    result_dict['boxes'].append(box)
                    result_dict['labels'].append(i+1)
                    # result_dict['scores'].append(obj[6])

        result_dict['boxes'] = np.array(result_dict['boxes'])
        
        return result_dict
                
class VideoCounting:
    def __init__(self, class_names, zone_path, minimum_length=4) -> None:
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.track_dict = [{} for i in range(self.num_classes)]
        self.minimum_length = minimum_length
        self.zone_path = zone_path
        self.polygons, self.directions = load_zone_anno(zone_path)
    
    def run(self, frames, tracks, labels, boxes, output_path=None):
        from .utils import color_list
        """
        obj id must starts from 0
        boxes in xyxy format
        """

        """
        self.track_dict stores:
            [
                {
                    track_id: {
                        points: [center_points],
                        frames: [frame_id],
                        color: color,

                    },...
                }
            ]
        """

        for (frame_id, track_id, label_id, box) in zip(frames, tracks, labels, boxes):
            
            if check_bbox_intersect_polygon(self.polygons, box):
                # check only boxes which intersect with polygons

                if track_id not in self.track_dict[label_id].keys():
                    self.track_dict[label_id][track_id] = {
                        'boxes': [],
                        'frames': [],
                        'color': random.sample(color_list,1),
                    }
                
                self.track_dict[label_id][track_id]['boxes'].append(box)
                self.track_dict[label_id][track_id]['frames'].append(frame_id)
                    
        
        for label_id in range(self.num_classes):
            for track_id in self.track_dict[label_id].keys():
                boxes = self.track_dict[label_id][track_id]['boxes']

                first_box = boxes[0]
                last_box = boxes[-1]

                center_point_first = ((first_box[2]+first_box[0]) / 2, (first_box[3] + first_box[1])/2)
                center_point_last = ((last_box[2]+last_box[0]) / 2, (last_box[3] + last_box[1])/2)

                direction = find_best_match_direction(
                    obj_vector = (center_point_first, center_point_last),
                    paths = self.directions
                )   

                self.track_dict[label_id][track_id]['direction'] = direction
        
        if output_path is not None:
            save_tracking_to_csv(self.track_dict, output_path)

        return self.track_dict
       



class Pipeline:
    def __init__(self, args, config, cam_config):
        self.detector = VideoDetect(args, config)
        self.class_names = self.detector.class_names
        self.video_path = args.input_path
        self.saved_path = args.output_path
        self.cam_config = cam_config
        self.zone_path = cam_config.zone_path
        self.config = config

        if os.path.isdir(self.video_path):
            video_names = sorted(os.listdir(self.video_path))
            self.all_video_paths = [os.path.join(self.video_path, i) for i in video_names]
        else:
            self.all_video_paths = [self.video_path]

    def get_cam_name(self, path):
        filename = os.path.basename(path)
        cam_name = filename[:-4]
        return cam_name

    def run(self):
        for video_path in self.all_video_paths:
            cam_name = self.get_cam_name(video_path)
            videoloader = VideoLoader(self.config, video_path)
            self.tracker = VideoTracker(
                len(self.class_names),
                self.cam_config.cam[cam_name],
                videoloader.dataset.video_info,
                deepsort_chepoint=self.cam_config.checkpoint)

            videowriter = VideoWriter(
                videoloader.dataset.video_info,
                saved_path=self.saved_path,
                obj_list=self.class_names)
            
            videocounter = VideoCounting(
                class_names = self.class_names,
                zone_path = os.path.join(self.zone_path, cam_name+".json"))

            obj_dict = {
                'frames': [],
                'tracks': [],
                'labels': [],
                'boxes': []
            }

            for idx, batch in enumerate(tqdm(videoloader)):
                preds = self.detector.run(batch)
                ori_imgs = batch['ori_imgs']

                for i in range(len(ori_imgs)):
                    boxes = preds['boxes'][i]
                    labels = preds['labels'][i]
                    scores = preds['scores'][i]
                    frame_id = batch['frames'][i]

                    ori_img = ori_imgs[i]
                    track_result = self.tracker.run(ori_img, boxes, labels, scores)
                    

                    # box_xywh = change_box_order(track_result['boxes'],'xyxy2xywh');
                    # videowriter.write(
                    #     ori_img,
                    #     boxes = track_result['boxes'],
                    #     labels = box_xywh,
                    #     tracks = track_result['tracks'])
                    
                    for j in range(len(track_result['boxes'])):
                        obj_dict['frames'].append(frame_id)
                        obj_dict['tracks'].append(track_result['tracks'][j])
                        obj_dict['labels'].append(track_result['labels'][j])
                        obj_dict['boxes'].append(track_result['boxes'][j])

            

            result_dict = videocounter.run(
                    frames = obj_dict['frames'],
                    tracks = obj_dict['tracks'], 
                    labels = obj_dict['labels'],
                    boxes = obj_dict['boxes'],
                    output_path=os.path.join(self.saved_path, cam_name+'.csv'))

            videoloader.reinitialize_stream()
            videowriter.write_full_to_video(
                videoloader,
                csv_path=os.path.join(self.saved_path, cam_name+'.csv'),
                paths=videocounter.directions,
                polygons=videocounter.polygons)
            


def main(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    num_gpus = len(args.gpus.split(','))
    devices_info = get_devices_info(args.gpus)

    if os.path.isdir(args.input_path):
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)


    ## Print info
    print(config)
    print(f"Nubmer of gpus: {num_gpus}")
    print(devices_info)

    cam_config = Config(os.path.join('configs', 'cam_configs.yaml'))             
    pipeline = Pipeline(args, config, cam_config)
    pipeline.run()

if __name__ == '__main__':
    args = parser.parse_args() 

    ignore_keys = [
        'min_iou_val',
        'min_conf_val',
        'tta',
        'gpu_devices',
        'tta_ensemble_mode',
        'tta_conf_threshold',
        'tta_iou_threshold',
    ]

    config = get_config(args.weight, ignore_keys)
    if config is None:
        print("Config not found. Load configs from configs/configs.yaml")
        config = Config(os.path.join('configs','configs.yaml'))
    else:
        print("Load configs from weight")    

    main(args, config)
    