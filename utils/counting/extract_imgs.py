import cv2
import os
from os.path import join
from tqdm import tqdm
from pathlib import Path
# Opens the Video file
from glob import glob
import argparse


def extract(filename, out_dir):
    cap = cv2.VideoCapture(filename)
    i = 0
    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    print("video fps:", fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=frame_count)
    os.makedirs(out_dir, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(os.path.join(out_dir,f'{i:05}' + ".jpg"), frame)
        pbar.update(1)
        i += 1
    pbar.close()
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
    parser.add_argument('--video_path',  help='configuration cam file')
    parser.add_argument('--output_path', help='output path')        
    args = parser.parse_args()

    video_name = os.path.basename(args.video_path)
    video_name = video_name[:-4]
    output_path = os.path.join(args.output_path, video_name)
    extract(args.video_path, output_path)