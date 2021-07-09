import numpy as np
import cv2
import json
import random
from tqdm import tqdm
from .bb_polygon import *

def random_color():
    rgbl = [255, 0, 0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def draw_arrow(image, start_point, end_point, color):
    cv2.line(image, start_point, end_point, color, 3)
    cv2.circle(image, end_point, 15, color, -1)


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
        cv2.polylines(image, [polygon], True, colors[0], 5)
    if paths:
        for path, points in paths.items():
            draw_arrow(image, points[0], points[1], colors[5])


def draw_start_last_points(ori_im, start_point, last_point, color=(0, 255, 0)):
    draw_arrow(ori_im, start_point, last_point, color)


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
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y), font,
                fontScale=font_scale, color=(0, 0, 0), thickness=1)
    return img


def get_dict(lines):
    draw_dict = {}
    for line in (lines):
        if len(line.split()) < 7:
            continue  # skip all null lines
        try:
            frame_id, move_id, obj_id, cls_id, xmin, ymin, xmax, ymax, start_point_x, start_point_y, last_point_x, last_point_y = list(
                map(int, line.split()))
        except Exception as e:
            print(e)
            break
        if frame_id not in draw_dict.keys():
            draw_dict[frame_id] = [
                (xmin, ymin, xmax, ymax, start_point_x, start_point_y, last_point_x, last_point_y, move_id, obj_id, cls_id)]
        else:
            draw_dict[frame_id].append(
                (xmin, ymin, xmax, ymax, start_point_x, start_point_y, last_point_x, last_point_y, move_id, obj_id, cls_id))
    return draw_dict


def visualize_merged(
    videoloader,
    outvid,
    flatten_db, 
    polygons_first, 
    polygons_last, 
    paths, 
    polygons):

    colors = [(0, 0, 255),  # red    0
              (0, 255, 0),  # green  1
              (255, 0, 0),  # blue   2
              (0, 255, 255),  # cyan   3
              (128, 0, 128),  # purple 4
              (0, 0, 0),  # black  5
              (255, 255, 255)]  # white  6

    # {label_id: {track_id: (move_id, start_frame, end_frame)}}
    track_db = {}  
    total_count = {}
    frame_count = {}
    for key in list(paths.keys()):
        key = int(key)
        total_count[key] = 0
        frame_count[key] = 0
    
    for batch in videoloader:
        ori_imgs = batch['ori_imgs']
        frame_ids = batch['frames']
        for (ori_img, frame_id) in zip(ori_imgs, frame_ids):
            ori_im = ori_img.copy()
            text = ''
            for key in list(total_count.keys()):
                total_count[key] += frame_count[key]
                frame_count[key] = 0
                text = text + '| direction{}: {}'.format(key, total_count[key])
            try:
                draw_text(ori_im, text)
            except:
                pass

            for label, polygon in polygons.items():
                # draw_anno(ori_im, polygon, paths)
                draw_anno(ori_im, polygon)

            for label, trigger in polygons_last.items():
                polygon = np.array(trigger, np.int32)
                polygon = polygon.reshape((-1, 1, 2))
                cv2.polylines(ori_im, [polygon], True, (255, 255, 0), 3)

            for label, trigger in polygons_first.items():
                polygon = np.array(trigger, np.int32)
                polygon = polygon.reshape((-1, 1, 2))
                cv2.polylines(ori_im, [polygon], True, (255, 255, 255), 3)

            db_index = 0
            while True:
                if not(db_index < len(flatten_db)):
                    break
                db_frame_id, vehicle_id, movement_id, obj_id, xmin, ymin, xmax, ymax, start_frame, last_frame = [
                    int(x) for x in flatten_db[db_index]]
                if frame_id != db_frame_id:
                    break
                bbox_xyxy = (xmin, ymin, xmax, ymax)
                ori_im = draw_bbox(ori_im, bbox_xyxy,
                                vehicle_id, movement_id, obj_id)

                if vehicle_id not in track_db.keys():
                    track_db[vehicle_id] = {}
                if obj_id not in track_db[vehicle_id].keys():
                    start_point_x = (xmax + xmin) // 2
                    start_point_y = (ymax + ymin) // 2
                    start_point = (start_point_x, start_point_y)
                    track_db[vehicle_id][obj_id] = (
                        movement_id, start_point, last_frame)
                else:
                    movement_id, start_point, last_frame = track_db[vehicle_id][obj_id]

                last_point_x = (xmax + xmin) // 2
                last_point_y = (ymax + ymin) // 2
                last_point = (last_point_x, last_point_y)
                draw_start_last_points(ori_im, start_point,
                                    last_point, colors[movement_id % 7])
                if start_frame == frame_id:
                    frame_count[movement_id] += 1
                if last_frame == frame_id:
                    del track_db[vehicle_id][obj_id]
                db_index += 1

            outvid.write(ori_im)

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255,  0), (0, 128,  0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139,  0,
                                                               139), (100, 149, 237), (138, 43, 226), (238, 130, 238),
             (255,  0, 255), (0, 100,  0), (127, 255,  0), (255,  0,
                                                            255), (0,  0, 205), (255, 140,  0), (255, 239, 213),
             (199, 21, 133), (124, 252,  0), (147, 112, 219), (106, 90,
                                                               205), (176, 196, 222), (65, 105, 225), (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199,
                                                               21, 133), (148,  0, 211), (255, 99, 71), (144, 238, 144),
             (255, 255,  0), (230, 230, 250), (0,  0, 255), (128, 128,
                                                             0), (189, 183, 107), (255, 255, 224), (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128,
                                                               128), (72, 209, 204), (139, 69, 19), (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135,
                                                               206, 235), (0, 191, 255), (176, 224, 230), (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139,
                                                                 139), (143, 188, 143), (255,  0,  0), (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42,
                                                              42), (178, 34, 34), (175, 238, 238), (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def draw_bbox(img, box, cls_name, move_id, identity=None, offset=(0, 0)):
    '''
        draw box of an id
    '''
    x1, y1, x2, y2 = [int(i+offset[idx % 2]) for idx, i in enumerate(box)]
    # set color and label text
    color = COLORS_10[identity %
                      len(COLORS_10)] if identity is not None else COLORS_10[0]
    label = 'c:{} m:{} id:{}'.format(cls_name, move_id, identity)
    # box text and bar
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
    cv2.putText(img, label, (x1, y1+t_size[1]+4),
                cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
    return img


def draw_bboxes(img, bbox, identities=None, cls=1, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label1 = 'c: {} o: {}'.format(cls, id)
        t_size = cv2.getTextSize(label1, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        cv2.rectangle(
            img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
        cv2.putText(
            img, label1, (x1, y1+t_size[1]+4), cv2. FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img


def softmax(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(x*5)
    return x_exp/x_exp.sum()


def softmin(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(-x)
    return x_exp/x_exp.sum()

def load_zone_anno(json_filename):
    """
    Load the json with ROI and MOI annotation.

    """
    with open(json_filename) as jsonfile:
        dd = json.load(jsonfile)
        polygons_first = {}
        polygons_last = {}
        paths = {}
        masks = {}
        for obj in dd['shapes']:
            if obj['label'][:4] == 'zone':
                idx = obj['label'][5:]
                polygon = [(int(x), int(y)) for x, y in obj['points']]
                if obj['label'][4] == '1':  # => dau ra
                    polygons_first[idx] = polygon
                    print(obj['label'], idx)
                elif obj['label'][4] == '2':  # => dau vao
                    polygons_last[idx] = polygon
            elif obj['label'][:9] == 'direction':
                kk = str(int(obj['label'][-2:]))
                paths[kk] = [(int(x), int(y)) for x, y
                             in obj['points']]
            if obj['label'][:4] == 'mask':
                ls_id = obj['label'].split('_')[1:]
                mask = [(int(x), int(y)) for x, y in obj['points']]
                for idx in ls_id:
                    masks[int(idx)] = mask
    return polygons_first, polygons_last, paths, masks

def counting_moi(polypack, paths, vehicle_vector_list, class_id):
    """
    Args:
      paths: List of MOI - (first_point, last_point)
      vehicle_vector_list: List of tuples (first_point, last_point, last_frame_id, track_id, bbox start, bbox last)

    Returns:
      A list of tuples (frame_id, movement_id, vehicle_class_id)
      A dict of tuples (frame_id, movement_id, vehicle_class_id)
    """
    moi_detection_list = []
    moi_detection_dict = {}
    movement_label = ''
    polygons_first, polygons_last = polypack
    # for vehicle_vector in vehicle_vector_list[:5]:

    for vehicle_vector in vehicle_vector_list:
        max_cosin = -2
        movement_id = ''
        last_frame = 0
        cosin = 0
        labels = []
        for label, polygon in polygons_last.items():
            labels = label.split('_')
            if check_bbox_intersect_polygon(polygon, vehicle_vector[5]):
                break

        for label2, polygon2 in polygons_first.items():
            labels2 = label2.split('_')

            if is_point_in_polygon(polygon2, vehicle_vector[0]):

                # if vehicle_vector[3] == 11:
                #     import pdb
                #     pdb.set_trace()
                for i in range(len(labels)):
                    if not(labels[i] in labels2):
                        labels[i] = ''
                break

        for movement_label, movement_vector in paths.items():
            if not(movement_label in labels):
                continue
            cosin = cosin_similarity(movement_vector, vehicle_vector)
            if cosin > max_cosin:
                max_cosin = cosin
                movement_id = movement_label
                last_frame = vehicle_vector[2]
        # print('==============')
        # if max_cosin < 0.5 or np.isnan(cosin):
        if np.isnan(cosin):
            continue
        if movement_id == '':
            movement_id = '0'
        moi_detection_dict[vehicle_vector[3]] = (
            movement_id, vehicle_vector[:2])
        moi_detection_list.append((last_frame, movement_id, class_id))
    return moi_detection_list, moi_detection_dict

def run_plan_in(draw_dict, polygons):
    count_database = {}
    for frame_id in tqdm(draw_dict.keys()):
        ls = draw_dict[frame_id]
        for box in ls:
            xmin, ymin, xmax, ymax, start_point_x, start_point_y, last_point_x, last_point_y, move_id, track_id, label_id = [
                int(i) for i in box]

            start_point = (start_point_x, start_point_y)
            last_point = (last_point_x, last_point_y)
            for _, polygon in polygons.items():
                polygon_edges = [(polygon[i], polygon[(i+1) % len(polygon)])
                                    for i in range(len(polygon)-1)]
                flag_intersect = False
                bbox_tmp = [(x, y) for x in [xmin, xmax]
                            for y in [ymin, ymax]]
                flag_intersect = is_bounding_box_intersect(
                    bbox_tmp, polygon)
                for edge in polygon_edges:
                    if is_intersect(edge[0], edge[1], start_point, last_point):
                        flag_intersect = True
                        break
                if not(flag_intersect):
                    continue
                key_database = f'{label_id}_{move_id}'
                if key_database not in count_database:
                    count_database[key_database] = {}
                if not is_point_in_polygon(polygon, start_point) and flag_intersect:
                    if track_id not in count_database[key_database].keys():
                        count_database[key_database][track_id] = []
                    count_database[key_database][track_id].append({
                        'frame_id': frame_id,
                        'xmin': xmin,
                        'xmax': xmax,
                        'ymin': ymin,
                        'ymax': ymax,
                    })

    
    return count_database