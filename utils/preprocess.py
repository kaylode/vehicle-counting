import cv2
from tqdm import tqdm
import argparse
import os


def video2frame(inp_path, out_path,output_size):
    path = inp_path
    root = out_path
    size = (output_size,output_size)
    video_name = path[-10:-4]
    out_dir = os.path.join(root, video_name)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    vidcap = cv2.VideoCapture(path)
    success,image = vidcap.read()
    count = 0
    success = True
    for i in tqdm(range(13500)):
        if not success:
            break
        success,image = vidcap.read()
        
        frame_name = str(count).zfill(5)+'.jpg'
        out_path = os.path.join(root, video_name,frame_name)
        #image = cv2.resize(image,size,interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(out_path, image)     # save frame as JPEG file
        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break
        count += 1

def main(args):
    videos_path = args.path
    output_path = args.out
    output_size = args.output_size

    videos_path = [os.path.join(videos_path,i) for i in os.listdir(videos_path) if i.endswith('.mp4')]
    for video in videos_path:
        video2frame(video,output_path,output_size)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split video into frame')
    parser.add_argument('--path', type=str, 
                        help='path to folder video .mp4')
    parser.add_argument('--out', type=str, 
                        help='path to folder frame output')
    parser.add_argument('--output_size', type=int, default= 768, 
                        help='size of output')                 
    args = parser.parse_args() 





    main(args)