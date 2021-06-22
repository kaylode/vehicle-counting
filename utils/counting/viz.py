import json
import math
import os
import random
import sys

import cv2
import numpy as np
from tqdm import tqdm

import bb_polygon
from bb_polygon import (is_bounding_box_intersect, is_intersect,
                        is_point_in_polygon)
from utils import draw_bbox


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
    vdo = [os.path.join(inpath, x) for x in os.listdir(inpath)]
    vdo.sort()
#     vdo = vdo[:]
    if debug:
        vdo = vdo[:3000]
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


if __name__ == "__main__":
    cam_id = '10'
    imgs_path = f'./cam_{cam_id}/'
    zone_path = f'./zones-movement_paths/cam_{cam_id}.json'
    json_anno_path = f'./zones-movement_paths/cam_{cam_id}.json'
    count_fuse_db = json.load(
        open('/home/ken/AIC/AIC-final/json/count_10.json'))
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
    visualize_merged(flatten_db,
                     imgs_path,
                     f'./visualize/cam_{cam_id}.mp4',
                     json_anno_path,
                     debug=True)
