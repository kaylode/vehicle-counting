import json
import math
import os
import random
import sys

import cv2
import numpy as np
from tqdm import tqdm
from utils.counting import bb_polygon

from utils.counting.bb_polygon import (is_bounding_box_intersect, is_intersect,
                        is_point_in_polygon)
from utils.counting.utils import draw_bbox
import argparse
from time import time


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


def check_bbox_intersect_polygon(polygon, bbox):
    """

    Args:
      polygon: List of points (x,y)
      bbox: A tuple (xmin, ymin, xmax, ymax)

    Returns:
      True if the bbox intersect the polygon
    """
    x1, y1, x2, y2 = bbox
    bb = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return bb_polygon.is_bounding_box_intersect(bb, polygon)


def cosin_similarity(a2d, b2d, debug=False):
    a = np.array((a2d[1][0] - a2d[0][0], a2d[1][1] - a2d[0][1])).astype(float)
    b = np.array((b2d[1][0] - b2d[0][0], b2d[1][1] - b2d[0][1])).astype(float)
    tmp1 = np.dot(a, b)
    tmp2 = (np.linalg.norm(a)*np.linalg.norm(b*1.0))
    if debug:
        print(a, b, tmp1, tmp2)
    return tmp1/tmp2


def counting_moi(polypack, paths, vehicle_vector_list, class_id, threshold=0.5):
    """
    Args:
      paths: List of MOI - (first_point, last_point)
      vehicle_vector_list: List of tuples (first_point, last_point, last_frame_id, obj_id, bbox start, bbox last)

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


def visualize_merged(flatten_db, inpath, outpath, json_anno_path, debug=False):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    vdo = [os.path.join(inpath, x) for x in os.listdir(inpath)]
    vdo.sort()
#     vdo = vdo[:]
    if debug:
        vdo = vdo[:300]
    colors = [(0, 0, 255),  # red    0
              (0, 255, 0),  # green  1
              (255, 0, 0),  # blue   2
              (0, 255, 255),  # cyan   3
              (128, 0, 128),  # purple 4
              (0, 0, 0),  # black  5
              (255, 255, 255)]  # white  6
    sample = cv2.imread(vdo[0])
    im_height = int(sample.shape[0])
    im_width = int(sample.shape[1])

    area = 0, 0, im_width, im_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(outpath, fourcc, 10, (im_width, im_height))
    pbar = tqdm(vdo)
    frame_id = 0
    tmp, triggers, paths, polygons = load_zone_anno(json_anno_path)
    db_index = 0
    track_db = {}  # {vehicle: {obj_id: (move_id, start_frame, end_frame)}}
    total_count = {}
    frame_count = {}
    for key in list(paths.keys()):
        key = int(key)
        total_count[key] = 0
        frame_count[key] = 0
    for path in pbar:
        frame_id += 1
        ori_im = cv2.imread(path)
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

        for label, trigger in triggers.items():
            polygon = np.array(trigger, np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            cv2.polylines(ori_im, [polygon], True, (255, 255, 0), 3)

        for label, trigger in tmp.items():
            polygon = np.array(trigger, np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            cv2.polylines(ori_im, [polygon], True, (255, 255, 255), 3)

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

        # frame_id += 1
        output.write(ori_im)
    output.release()


def run_plan_in(draw_dict, inpath, output_count_path, json_anno_path, debug=False):
    # vdo = [os.path.join(inpath, x) for x in os.listdir(inpath)]
    # vdo.sort()
    # vdo = vdo[:]
    # if debug:
    #     vdo = vdo[:3000]
    pbar = (range(18000))
    frame_id = 0
    _, _, paths, polygons = load_zone_anno(json_anno_path)

    count_database = {}
    for path in pbar:
        if frame_id in draw_dict.keys():
            ls = draw_dict[frame_id]
            for i, box in enumerate(ls):
                xmin, ymin, xmax, ymax, start_point_x, start_point_y, last_point_x, last_point_y, move_id, obj_id, cls_id = [
                    int(i) for i in box]
                bbox_xyxy = (xmin, ymin, xmax, ymax)

                start_point = (start_point_x, start_point_y)
                last_point = (last_point_x, last_point_y)
                for label, polygon in polygons.items():
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
                    key_database = f'{cls_id}_{move_id}'
                    if key_database not in count_database:
                        count_database[key_database] = {}
                    if not is_point_in_polygon(polygon, start_point) and flag_intersect:
                        if obj_id not in count_database[key_database].keys():
                            count_database[key_database][obj_id] = []
                        count_database[key_database][obj_id].append({
                            'frame_id': frame_id,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax,
                        })

        frame_id += 1
    # json.dump(count_database, open(output_count_path, 'wt'))
    return count_database


def unpack(line, unwarp=0):
    if unwarp == 1:  # if file LMQ
        frame_id, cls_id, obj_id, xmin, ymin, xmax, ymax = list(
            map(int, line.split()))  # file LMQ
        # cls_id += 1  # LMQ only
    else:  # file kento
        frame_id, obj_id, cls_id, xmin, ymin, xmax, ymax = list(
            map(int, line.split()))
    return frame_id, cls_id+1, obj_id, xmin, ymin, xmax, ymax


def run_all(args, cam_id, track_filename, viz=False, start=0):
    random.seed(0)

    imgs_path = args.img_folder
    tracking_path = args.tracking_path
    movement_path = f'./results/counting/{track_filename}_temp.txt'
    zone_path = args.zone_path
    json_anno_path = args.zone_path
    result_filename = os.path.join(args.output_path, track_filename, track_filename+'_count.txt')
    result_videoname = os.path.join(args.output_path, track_filename, track_filename+'_count.mp4')
    _, _, paths, polygons = load_zone_anno(zone_path)

    class_ids = [1, 2]
    track_dict_ls = [{}, {}, {}, {}, {}]
    with open(tracking_path) as f:
        lines = f.readlines()
    for line in (lines):
        if len(line.split()) < 7:
            print('SKIP!!')
            continue  # skip all null lines
        frame_id, cls_id, obj_id, xmin, ymin, xmax, ymax = unpack(line)

        for label, polygon in polygons.items():
            if check_bbox_intersect_polygon(polygon, (xmin, ymin, xmax, ymax)):
                if obj_id not in track_dict_ls[cls_id].keys():
                    # find obj id which intersect with polygons
                    track_dict_ls[cls_id][obj_id] = []

    for line in (lines):
        if len(line.split()) < 7:
            print('SKIP!!')
            continue  # skip all null lines
        frame_id, cls_id, obj_id, xmin, ymin, xmax, ymax = unpack(line)

        if obj_id in track_dict_ls[cls_id].keys():
            track_dict_ls[cls_id][obj_id].append(
                (xmin, ymin, xmax, ymax, frame_id))

    n = 0
    max_dis = -1
    for line in (lines):
        if len(line.split()) < 7:
            print('SKIP!!')
            continue  # skip all null lines
        # frame_id, obj_id, cls_id, xmin, ymin, xmax, ymax = list(map(int, line.split())) # file Kento
        frame_id, cls_id, obj_id, xmin, ymin, xmax, ymax = unpack(line)

        if obj_id in track_dict_ls[cls_id].keys():
            if len(track_dict_ls[cls_id][obj_id]) < 4:
                del track_dict_ls[cls_id][obj_id]

    # avg_distance /= n
    # print(max_dis)
    # print(avg_distance)

    vehicle_ls = [[], [], [], [], []]
    for class_id in class_ids:
        track_dict = track_dict_ls[class_id]
        for tracker_id, tracker_list in track_dict.items():
            if len(tracker_list) > 1:
                first = tracker_list[0]
                last = tracker_list[-1]
                first_point = ((first[2] + first[0])/2,
                               (first[3] + first[1])/2)
                last_point = ((last[2] + last[0])/2, (last[3] + last[1])/2)
                vehicle_ls[class_id].append(
                    (first_point, last_point, last[4], tracker_id, first[:4], last[:4]))

    polygons_first, polygons_last, paths, _ = load_zone_anno(zone_path)
    vehicles_moi_detections_ls = [[], [], [], [], []]
    vehicles_moi_detections_dict = [{}, {}, {}, {}, {}]
    for class_id in class_ids:
        vehicles_moi_detections_ls[class_id], vehicles_moi_detections_dict[class_id] = \
            counting_moi((polygons_first, polygons_last),
                         paths, vehicle_ls[class_id], class_id)
    # import pdb
    # pdb.set_trace()
    class_ids = [1, 2]

    with open(tracking_path) as f:
        lines = f.readlines()
    g = open(movement_path, 'w')
    PRINT_POINT = True
    unique = {'0'}
    for line in (lines):
        if len(line.split()) < 7:
            continue  # skip all null lines
        frame_id, cls_id, obj_id, xmin, ymin, xmax, ymax = unpack(line)

        if obj_id in vehicles_moi_detections_dict[cls_id].keys():
            mov_id = vehicles_moi_detections_dict[cls_id][obj_id][0]
            if mov_id == '0':
                continue
            unique.add(mov_id)
            start_point = vehicles_moi_detections_dict[cls_id][obj_id][1][0]
            last_point = vehicles_moi_detections_dict[cls_id][obj_id][1][1]
            if PRINT_POINT:
                start_point = list(map(int, start_point))
                last_point = list(map(int, last_point))
                g.write(
                    f'{frame_id} {mov_id} {obj_id} {cls_id} {xmin} {ymin} {xmax} {ymax} {start_point[0]} {start_point[1]} {last_point[0]} {last_point[1]}\n')
            else:
                g.write(
                    f'{frame_id} {mov_id} {obj_id} {cls_id} {xmin} {ymin} {xmax} {ymax}\n')

    g.close()

    with open(movement_path) as f:
        lines = f.readlines()
    draw_dict = get_dict(lines)
    count_fuse_db = run_plan_in(
        draw_dict, imgs_path, f'./json/count_{cam_id}.json', json_anno_path, False)

    
    video_id = f'cam_{cam_id}'
    _, _, paths, polygons = load_zone_anno(zone_path)

    video_mapping = {}
    with open('home/ubuntu/khoi-ws/data/Dataset_A/list_video_id.txt') as f:
        lines = f.readlines()
    
    for line in lines:
        tokens = line.split(' ')
        video_mapping[tokens[-1]] = int(tokens[0])

    with open(result_filename, 'w') as result_file, open(args.submission, 'a+') as submission_file:
        for key in count_fuse_db.keys():
            class_id, move_id = key.split('_')
            for obj, info in count_fuse_db[key].items():

                tmps = [(x['frame_id'], x['xmin'], x['xmax'],
                         x['ymin'], x['ymax']) for x in info]
                frame_id, xmin, xmax, ymin, ymax = tmps[-1]
                point = ((xmax+xmin)//2, (ymax+ymin)//2)
                for label, polygon in polygons.items():
                    flag = False
                    if check_bbox_intersect_polygon(polygon, (xmin, ymin, xmax, ymax)):
                        result_file.write('{} {} {} {}\n'.format(
                            video_id, frame_id, move_id, class_id))
                        break
                    else:
                        for i in range(1, len(tmps)-1):
                            frame_id, xmin, xmax, ymin, ymax = tmps[-1 * i]
                            point = ((xmax+xmin)//2, (ymax+ymin)//2)
                            if check_bbox_intersect_polygon(polygon, (xmin, ymin, xmax, ymax)):
                                result_file.write('{} {} {} {}\n'.format(
                                    video_id, frame_id, move_id, class_id))
                                running_time = time.time() - start

                                submission_file.write('{} {} {} {} {} {}\n'.format(
                                    running_time,
                                    video_mapping[video_id], frame_id, move_id, class_id))
                                flag = True
                                break
                        if flag:
                            break

    # merged: {'vehicle_id'_'movement_id': {obj_id: [{'frame_id': frame_id, 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}]}}
    merged = count_fuse_db
    flatten_db = []
    for key, value in merged.items():
        vehicle_id, movement_id = key.split('_')
        for obj_id, frame_list in value.items():
            for frame_info in frame_list:
                frame_id = frame_info['frame_id']
                xmin = frame_info['xmin']
                xmax = frame_info['xmax']
                ymin = frame_info['ymin']
                ymax = frame_info['ymax']
                start_frame = frame_list[0]['frame_id']
                last_frame = frame_list[-1]['frame_id']
                flatten_db.append((frame_id, vehicle_id, movement_id,
                                   obj_id, xmin, ymin, xmax, ymax, start_frame, last_frame))

    flatten_db = sorted(flatten_db, key=lambda x: x[0])
    if viz:
        visualize_merged(flatten_db,
                        imgs_path,
                        result_videoname,
                        json_anno_path,
                        debug=False)

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
    parser.add_argument('video_name',  help='configuration cam file')
    parser.add_argument('--img_folder', type=str, default=None,  help='configuration cam file')
    parser.add_argument('--tracking_path', type=str, default=None,  help='configuration cam file')
    parser.add_argument('--zone_path', type=str, default=None,  help='configuration cam file')
    parser.add_argument('--output_path', type=str, default=None,  help='configuration cam file')
    parser.add_argument('--viz', action='store_true',  help='configuration cam file')
    parser.add_argument('--submission', type=str, default='./results/submission.txt',  help='configuration cam file')
    args = parser.parse_args()

    with open('time.txt') as f:
        lines = f.readlines()

    total_time = 0
    for line in lines:
        tokens = line.split(' ')
        if tokens[0] == args.video_name:
            total_time = sum([float(i) for i in tokens[1:]])
        
    tokens = args.video_name.split('_')

    cam_id = tokens[-1] if tokens[-1].isnumeric() else tokens[-2]
    cam_id = int(cam_id)
    run_all(args, cam_id, args.video_name, args.viz, total_time+start)
    
