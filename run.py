from utils.getter import *
import argparse
import os
from modules import CountingPipeline

parser = argparse.ArgumentParser(description='Perfom Objet Detection')
parser.add_argument('--weight', type=str, default = None,help='version of EfficentDet')
parser.add_argument('--input_path', type=str, help='path to an image to inference')
parser.add_argument('--output_path', type=str, help='path to save inferenced image')
parser.add_argument('--gpus', type=str, default='0', help='path to save inferenced image')
parser.add_argument('--min_conf', type=float, default= 0.1, help='minimum confidence for an object to be detect')
parser.add_argument('--min_iou', type=float, default=0.5, help='minimum iou threshold for non max suppression')
parser.add_argument('--tta', action='store_true', help='whether to use test time augmentation')
parser.add_argument('--tta_ensemble_mode', type=str, default='wbf', help='tta ensemble mode')
parser.add_argument('--tta_conf_threshold', type=float, default=0.01, help='tta confidence score threshold')
parser.add_argument('--tta_iou_threshold', type=float, default=0.9, help='tta iou threshold')
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

    ignore_keys = [
        'min_iou_val',
        'min_conf_val',
        'tta',
        'gpu_devices',
        'tta_ensemble_mode',
        'tta_conf_threshold',
        'tta_iou_threshold',
    ]

    if args.weight is None:
        config = None
    else:
        config = get_config(args.weight, ignore_keys)
    if config is None:
        print("Config not found. Load configs from configs/configs.yaml")
        config = Config(os.path.join('configs','configs.yaml'))
    else:
        print("Load configs from weight")  

    # If you not use any weight and want to use pretrained on COCO, uncomment these lines
    if args.weight is None:
        MAPPING_DICT = {
            0: 0,
            1: 0,
            2: 1,
            3: 0,
            5: 2,
            7: 3
        }
        args.mapping_dict = MAPPING_DICT

    main(args, config)
    
