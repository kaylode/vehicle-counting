import json
import math
import os
import random
import sys

import cv2
import numpy as np
from tqdm import tqdm

from utils.counting import *
import argparse
from time import time


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
                xmin, ymin, xmax, ymax, start_point_x, start_point_y, last_point_x, last_point_y, move_id, track_id, label_id = [
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

        frame_id += 1
    # json.dump(count_database, open(output_count_path, 'wt'))
    return count_database


def unpack(line, unwarp=0):
    if unwarp == 1:  # if file LMQ
        frame_id, label_id, track_id, xmin, ymin, xmax, ymax = list(
            map(int, line.split()))  # file LMQ
        # label_id += 1  # LMQ only
    else:  # file kento
        frame_id, track_id, label_id, xmin, ymin, xmax, ymax = list(
            map(int, line.split()))
    return frame_id, label_id+1, track_id, xmin, ymin, xmax, ymax




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
        frame_id, label_id, track_id, xmin, ymin, xmax, ymax = unpack(line)

        for label, polygon in polygons.items():
            if check_bbox_intersect_polygon(polygon, (xmin, ymin, xmax, ymax)):
                if track_id not in track_dict_ls[label_id].keys():
                    # find obj id which intersect with polygons
                    track_dict_ls[label_id][track_id] = []
                

    for line in (lines):
        if len(line.split()) < 7:
            print('SKIP!!')
            continue  # skip all null lines
        frame_id, label_id, track_id, xmin, ymin, xmax, ymax = unpack(line)

        if track_id in track_dict_ls[label_id].keys():
            track_dict_ls[label_id][track_id].append(
                (xmin, ymin, xmax, ymax, frame_id))

    n = 0
    max_dis = -1
    for line in (lines):
        if len(line.split()) < 7:
            print('SKIP!!')
            continue  # skip all null lines
        # frame_id, track_id, label_id, xmin, ymin, xmax, ymax = list(map(int, line.split())) # file Kento
        frame_id, label_id, track_id, xmin, ymin, xmax, ymax = unpack(line)

        if track_id in track_dict_ls[label_id].keys():
            if len(track_dict_ls[label_id][track_id]) < 4:
                del track_dict_ls[label_id][track_id]

    # avg_distance /= n
    # print(max_dis)
    # print(avg_distance)

    vehicle_tracks = [[], [], [], [], []]
    for class_id in class_ids:
        track_dict = track_dict_ls[class_id]
        for tracker_id, tracker_list in track_dict.items():
            if len(tracker_list) > 1:
                first = tracker_list[0]
                last = tracker_list[-1]
                first_point = ((first[2] + first[0])/2,
                               (first[3] + first[1])/2)
                last_point = ((last[2] + last[0])/2, (last[3] + last[1])/2)
                vehicle_tracks[class_id].append(
                    (first_point, last_point, last[4], tracker_id, first[:4], last[:4]))

    polygons_first, polygons_last, paths, _ = load_zone_anno(zone_path)
    vehicles_moi_detections_ls = [[], [], [], [], []]
    vehicles_moi_detections_dict = [{}, {}, {}, {}, {}]
    for class_id in class_ids:
        vehicles_moi_detections_ls[class_id], vehicles_moi_detections_dict[class_id] = \
            counting_moi((polygons_first, polygons_last),
                         paths, vehicle_tracks[class_id], class_id)
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
        frame_id, label_id, track_id, xmin, ymin, xmax, ymax = unpack(line)

        if track_id in vehicles_moi_detections_dict[label_id].keys():
            mov_id = vehicles_moi_detections_dict[label_id][track_id][0]
            if mov_id == '0':
                continue
            unique.add(mov_id)
            start_point = vehicles_moi_detections_dict[label_id][track_id][1][0]
            last_point = vehicles_moi_detections_dict[label_id][track_id][1][1]
            if PRINT_POINT:
                start_point = list(map(int, start_point))
                last_point = list(map(int, last_point))
                g.write(
                    f'{frame_id} {mov_id} {track_id} {label_id} {xmin} {ymin} {xmax} {ymax} {start_point[0]} {start_point[1]} {last_point[0]} {last_point[1]}\n')
            else:
                g.write(
                    f'{frame_id} {mov_id} {track_id} {label_id} {xmin} {ymin} {xmax} {ymax}\n')

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

    # merged: {'vehicle_id'_'movement_id': {track_id: [{'frame_id': frame_id, 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}]}}
    merged = count_fuse_db
    flatten_db = []
    for key, value in merged.items():
        vehicle_id, movement_id = key.split('_')
        for track_id, frame_list in value.items():
            for frame_info in frame_list:
                frame_id = frame_info['frame_id']
                xmin = frame_info['xmin']
                xmax = frame_info['xmax']
                ymin = frame_info['ymin']
                ymax = frame_info['ymax']
                start_frame = frame_list[0]['frame_id']
                last_frame = frame_list[-1]['frame_id']
                flatten_db.append((frame_id, vehicle_id, movement_id,
                                   track_id, xmin, ymin, xmax, ymax, start_frame, last_frame))

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
    
