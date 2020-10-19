import cv2
from tqdm import tqdm
import argparse
import os
import numpy as np
from decord import VideoReader
from decord import cpu, gpu

def main(args):
    path = args.path
    root = args.out

    video_name = path[-10:-4]
    out_dir = os.path.join(root, video_name)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    vidcap = cv2.VideoCapture(path)
    
    count = 0
    success = True
    for i in tqdm(range(13500)):
        if not success:
            break
        success,image = vidcap.read()
        
        frame_name = str(count).zfill(5)+'.jpg'
        out_path = os.path.join(root, video_name,frame_name)

        cv2.imwrite(out_path, image)     # save frame as JPEG file
        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break
        count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split video into frame')
    parser.add_argument('--path', type=str, 
                        help='path to video .mp4')
    parser.add_argument('--out', type=str, 
                        help='path to frame output')                 
    args = parser.parse_args()                    
    main(args)
  