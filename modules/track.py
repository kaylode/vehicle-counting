import numpy as np
import random
from networks import DeepSort
from utilities.counting import (
    load_zone_anno, check_bbox_intersect_polygon, 
    find_best_match_direction, save_tracking_to_csv)

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
        labels__ = labels.copy()
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
                    result_dict['labels'].append(i)
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
        from utilities.utils import color_list
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
                        'color': random.sample(color_list,1)[0],
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