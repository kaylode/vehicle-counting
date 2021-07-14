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
    image = cv2.line(image, start_point, end_point, color, 3)
    image = cv2.circle(image, end_point, 15, color, -1)
    return image

def draw_start_last_points(ori_im, start_point, last_point, color=(0, 255, 0)):
    return draw_arrow(ori_im, start_point, last_point, color)

def draw_one_box(img, box, key=None, value=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness

    coord = [box[0], box[1], box[2], box[3]]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    img = cv2.rectangle(img, c1, c2, color, thickness=tl*2)
    if key is not None and value is not None:
        header = f'{key} || {value}'
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(f'| {value}', 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(f'{key} |', 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        img = cv2.rectangle(img, c1, c2, color, -1)  # filled
        img = cv2.putText(img, header, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)    
    return img

def draw_text(img, text):
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    # set the rectangle background to white
    rectangle_bgr = (255, 255, 255)
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=1)[0]
    # set the text start position
    text_offset_x = 10
    text_offset_y = img.shape[0] - 25
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x +
                                                   text_width + 2, text_offset_y - text_height - 2))
    img = cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    img = cv2.putText(img, text, (text_offset_x, text_offset_y), font,
                fontScale=font_scale, color=(0, 0, 0), thickness=1)
    return img

def draw_anno(image, polygon=None, paths=None):
    colors = [(0, 0, 255),  # red    0
              (0, 255, 0),  # green  1
              (255, 0, 0),  # blue   2
              (0, 255, 255),  # cyan   3
              (128, 0, 128),  # purple 4
              (0, 0, 0),  # black  5
              (255, 255, 255)]  # white  6
    if polygon:
        polygon = np.array(polygon, np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        image = cv2.polylines(image, [polygon], True, colors[0], 5)
    if paths:
        for path, points in paths.items():
            points = np.array(points, np.int32)
            image = draw_arrow(image, points[0], points[1], colors[5])
    return image

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
    return '01'

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

    df['direction']= df['direction'].astype(str)
    
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

def visualize_one_frame(img, df):
    # track_id	frame_id	box	color	label	direction	fpoint	lpoint	fframe	lframe
    anns = [
        i for i in zip(
            df.track_id, 
            df.box, 
            df.color, 
            df.label,
            df.fpoint)
    ]

    for (track_id, box, color, label, fpoint) in anns:
        box = eval(box)
        fpoint = np.array(eval(fpoint)).astype(int)
        color = eval(color)
        cpoint = np.array([(box[2]+box[0]) / 2, (box[3] + box[1])/2]).astype(int)
        img = draw_one_box(
                img, 
                box, 
                key=f'id: {track_id}',
                value=f'cls: {label}',
                color=color)
        img = draw_start_last_points(img, fpoint, cpoint, color)
        
    return img

def count_frame_directions(df, count_dict):
    anns = [
        i for i in zip(
            df.frame_id,
            df.label, 
            df.direction,  
            df.lframe)
    ]

    for (frame_id, label, direction, lframe) in anns:
        if lframe == frame_id:
            count_dict[direction][label] += 1

    count_text = ""
    for dir in count_dict.keys():
        tmp_text = f"direction: {dir} | "
        for cls_id in count_dict[dir].keys():
            tmp_text += f"{cls_id}: {count_dict[dir][cls_id]} "
        count_text = count_text + tmp_text + '\n'

    return count_dict, count_text

def visualize_merged(videoloader, csv_path, directions, zones, num_classes, outvid):
    df = pd.read_csv(csv_path)
    count_dict = {
        dir: {
            label: 0 for label in range(num_classes)
        } for dir in directions
    }

    for batch in tqdm(videoloader):
        imgs = batch['ori_imgs']
        frame_ids = batch['frames']

        for idx in range(len(imgs)):
            frame_id = frame_ids[idx]
            img = imgs[idx].copy()

            tmp_df = df[df.frame_id.astype(int) == frame_id]
            count_dict, text = count_frame_directions(tmp_df, count_dict)

            img = draw_anno(img, zones, directions)

            if len(tmp_df) > 0:
                img = visualize_one_frame(img, tmp_df)
                
            img = draw_text(img, text)
            
            outvid.write(img)
