from utilities.getter import *
import argparse
import os
from modules import CountingPipeline

parser = argparse.ArgumentParser(description='Perform Counting vehicles')
parser.add_argument('--weight', type=str, default = None,help='checkpoint of yolo')
parser.add_argument('--input_path', type=str, help='path to an image to inference')
parser.add_argument('--output_path', type=str, help='path to save inferenced image')
parser.add_argument('--gpus', type=str, default='0', help='path to save inferenced image')
parser.add_argument('--debug', action='store_true', help='save detection at')
parser.add_argument('--mapping', default=None, help='Specify a class mapping if using pretrained')

def main(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    num_gpus = len(args.gpus.split(','))
    devices_info = get_devices_info(args.gpus)

    if os.path.isdir(args.input_path):
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

    ## Print info
    print(config)
    print(f"Nubmer of gpus: {num_gpus}")
    print(devices_info)

    cam_config = Config(os.path.join('configs', 'cam_configs.yaml'))             
    pipeline = CountingPipeline(args, config, cam_config)
    pipeline.run()

if __name__ == '__main__':
    args = parser.parse_args() 
    config = Config(os.path.join('configs','configs.yaml'))


    # If you not use any weight and want to use pretrained on COCO, uncomment these lines
    MAPPING_DICT = {
        0: 0,
        1: 0,
        2: 1,
        3: 0,
        5: 2,
        7: 3
    }
    args.mapping_dict = None #MAPPING_DICT

    main(args, config)
    
