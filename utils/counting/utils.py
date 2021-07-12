import cv2
import json
import random
from tqdm import tqdm
import pandas as pd
from .bb_polygon import *

def random_color():
    rgbl = [255, 0, 0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def draw_arrow(image, start_point, end_point, color):
    cv2.line(image, start_point, end_point, color, 3)
    cv2.circle(image, end_point, 15, color, -1)

def draw_start_last_points(ori_im, start_point, last_point, color=(0, 255, 0)):
    draw_arrow(ori_im, start_point, last_point, color)

def load_zone_anno(zone_path):
        with open(zone_path, 'r') as f:
            anno = json.load(f)
        
        directions =  {}
        zone = anno['shapes'][0]['points']
        for i in anno['shapes']:
            if i['label'].startswith('direction'):
                directions[i['label'][-2:]] = i['points']
        return zone, directions

def find_best_match_direction(obj_vector,paths):
    return 0

def save_tracking_to_csv(track_dict, filename):
    num_classes = len(track_dict)
    obj_dict = {
        'track_id': [],
        'frame_id': [],
        'box': [],
        'color': [],
        'label': [],
        'direction': [],
        'fpoint': [],
        'lpoint': [],
        'fframe': [],
        'lframe': []
    }

    for label_id in range(num_classes):
        for track_id in track_dict[label_id].keys():
            direction = track_dict[label_id][track_id]['direction']
            boxes = track_dict[label_id][track_id]['boxes']
            frames = track_dict[label_id][track_id]['frames']
            color = track_dict[label_id][track_id]['color']

            frame_first = frames[0]
            frame_last = frames[-1]

            box_first = boxes[0]
            box_last = boxes[-1]

            center_point_first = ((box_first[2]+box_first[0]) / 2, (box_first[3] + box_first[1])/2)
            center_point_last = ((box_last[2]+box_last[0]) / 2, (box_last[3] + box_last[1])/2)

            for i in range(len(track_dict[label_id][track_id]['boxes'])):              
                obj_dict['track_id'].append(track_id)
                obj_dict['frame_id'].append(frames[i])
                obj_dict['box'].append(boxes[i].tolist())
                obj_dict['color'].append(color)
                obj_dict['label'].append(label_id)
                obj_dict['direction'].append(direction)
                obj_dict['fpoint'].append(center_point_first)
                obj_dict['lpoint'].append(center_point_last)
                obj_dict['fframe'].append(frame_first)
                obj_dict['lframe'].append(frame_last)

    df = pd.DataFrame(obj_dict)
    df.to_csv(filename, index=False)


def convert_frame_dict(track_dict):
    """
    return result dict:
    {
        frame_id: {
            'boxes': [],
            'colors': [],
            'fpoints': [],
            'lpoints': []
        }
    }
    """
    result_dict = {}
    num_classes = len(track_dict)
    for label_id in range(num_classes):
        for track_id in track_dict[label_id].keys():
            direction = track_dict[label_id][track_id]['direction']
            boxes = track_dict[label_id][track_id]['boxes']
            frames = track_dict[label_id][track_id]['frames']
            color = track_dict[label_id][track_id]['color']

            for i in range(len(track_dict[label_id][track_id])):
                frame_id = frames[i]
                box = boxes[i]
                
                if frame_id not in result_dict.keys():
                    result_dict[frame_id] = {
                        'boxes': [],
                        'colors': [],
                        'fpoints': [],
                        'lpoints': [],
                        'labels': [],
                        'directions': []
                    }

                first_box = box[0]
                last_box = box[-1]
                center_point_first = ((first_box[2]+first_box[0]) / 2, (first_box[3] + first_box[1])/2)
                center_point_last = ((last_box[2]+last_box[0]) / 2, (last_box[3] + last_box[1])/2)

                result_dict[frame_id]['boxes'].append(box)
                result_dict[frame_id]['fpoints'].append(center_point_first)
                result_dict[frame_id]['lpoints'].append(center_point_last)
                result_dict[frame_id]['directions'].append(direction)
                result_dict[frame_id]['colors'].append(color)
                result_dict[frame_id]['labels'].append(label_id)

    return result_dict

def visualize_merged(frame_dict, polygons, paths, num_classes):
    num_frames = frame_dict.keys()

    direction_label_count = {
        k: {
            k2:0 for k2 in range(num_classes)
        } for k in paths
    }

    for frame_id in num_frames:
        frame_objects = frame_dict[frame_id]
        num_objs_in_frame = len(frame_objects)
        for index in range(num_objs_in_frame):
            box = frame_objects['boxes'][index]
            label = frame_objects['labels'][index]
            color = frame_objects['colors'][index]
            direction = frame_objects['directions'][index]
            direction_label_count[direction][label]+=1