from utils.getter import *
import argparse
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils import draw_boxes_v2, write_to_video
from utils.postprocess import postprocessing
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentations.transforms import get_resize_augmentation
from augmentations.transforms import MEAN, STD

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
parser.add_argument('--viz', action='store_true', help='save detection at')


class Testset():
    def __init__(self, config, input_path):
        self.input_path = input_path # path to video file
        self.image_size = config.image_size
        self.get_video_info()
        self.transforms = A.Compose([
            get_resize_augmentation(config.image_size, keep_ratio=config.keep_ratio),
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
            ToTensorV2(p=1.0)
        ])

        self.current_stream = None
        self.current_frame_id = 0

    def get_batch_size(self):
        # Temporary
        return 1

    def get_video_info(self):
        self.video_info = {}
        self.num_frames = 0

        vidcap = cv2.VideoCapture(self.input_path)
        video_name = os.path.basename(self.input_path)
        if vidcap.isOpened(): 
            # get vidcap property 
            WIDTH  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
            HEIGHT = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
            FPS = int(vidcap.get(cv2.CAP_PROP_FPS))
            NUM_FRAMES = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_info = {
                'name': video_name,
                'width': WIDTH,
                'height': HEIGHT,
                'fps': FPS,
                'num_frames': NUM_FRAMES
            }
            self.num_frames += NUM_FRAMES
        else:
            raise f"Cannot read video {video_name}"
    
    def init_reader(self):
        self.current_stream = cv2.VideoCapture(self.input_path)
        self.current_frame_id += 1

    def __getitem__(self, idx):
        success, ori_frame = self.current_stream.read()
        if not success:
            print(f"Cannot read frame {self.current_frame_id} from {self.video_info['name']}")
            return None
        else:
            self.current_frame_id+=1
        frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame /= 255.0
        if self.transforms is not None:
            inputs = self.transforms(image=frame)['image']

        image_w, image_h = self.image_size
        ori_height, ori_width, _ = ori_frame.shape

        return {
            'img': inputs,
            'ori_img': ori_frame,
            'image_ori_w': ori_width,
            'image_ori_h': ori_height,
            'image_w': image_w,
            'image_h': image_h,
        }

    def collate_fn(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        imgs = torch.stack([s['img'] for s in batch])   
        ori_imgs = [s['ori_img'] for s in batch]
        image_ori_ws = [s['image_ori_w'] for s in batch]
        image_ori_hs = [s['image_ori_h'] for s in batch]
        image_ws = [s['image_w'] for s in batch]
        image_hs = [s['image_h'] for s in batch]
        img_scales = torch.tensor([1.0]*len(batch), dtype=torch.float)
        img_sizes = torch.tensor([imgs[0].shape[-2:]]*len(batch), dtype=torch.float)   

        return {
            'imgs': imgs,
            'ori_imgs': ori_imgs,
            'image_ori_ws': image_ori_ws,
            'image_ori_hs': image_ori_hs,
            'image_ws': image_ws,
            'image_hs': image_hs,
            'img_sizes': img_sizes, 
            'img_scales': img_scales
        }

    def __len__(self):
        return self.num_frames

    def __str__(self):
        s2 = f"Number of frames: {self.num_frames}"
        return s2

class VideoDetect:
    def __init__(self, args, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')   
        self.video_path = args.input_path
        self.saved_path = args.output_path
        self.viz = args.viz
        self.config = config
        self.min_iou = args.min_iou
        self.min_conf = args.min_conf
        self.max_dets=config.max_post_nms
        self.keep_ratio=config.keep_ratio
        self.fusion_mode=config.fusion_mode

        if args.tta:
            self.tta = TTA(
                min_conf=args.tta_conf_threshold, 
                min_iou=args.tta_iou_threshold, 
                postprocess_mode=args.tta_ensemble_mode)
        else:
            self.tta = None

        if args.weight is not None:
            self.class_names, num_classes = get_class_names(args.weight)
        self.class_names.insert(0, 'Background')

        net = get_model(
            args, config, 
            self.device, 
            num_classes=num_classes)

        self.model = Detector(model = net, device = self.device)
        self.model.eval()

        if args.weight is not None:                
            load_checkpoint(self.model, args.weight)

        self.load_videos()

    def load_videos(self):
        self.all_video_paths = []   
        if os.path.isdir(self.video_path):  # path to video folder
            paths = sorted(os.listdir(self.video_path))
            for path in paths:
                self.all_video_paths.append(os.path.join(self.video_path, path))
        elif os.path.isfile(self.video_path): # path to single video
            self.all_video_paths.append(self.video_path)
        self.num_videos = len(self.all_video_paths)

    def init_loader(self, idx):
        video_path = self.all_video_paths[idx]
        self.dataset = Testset(self.config, video_path)
        self.dataset.init_reader()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size= self.dataset.get_batch_size(),
            num_workers=0,
            pin_memory=True,
            collate_fn= self.dataset.collate_fn)

    def init_writer(self):
        video_name = self.dataset.video_info['name']
        outpath =os.path.join(self.saved_path, video_name)
        FPS = self.dataset.video_info['fps']
        WIDTH = self.dataset.video_info['width']
        HEIGHT = self.dataset.video_info['height']
        NUM_FRAMES = self.dataset.video_info['num_frames']
        self.outvid = cv2.VideoWriter(
            outpath,   
            cv2.VideoWriter_fourcc(*'mp4v'), 
            FPS, 
            (WIDTH, HEIGHT)
        )

    def run(self):
        for video_idx in range(self.num_videos):
            self.init_loader(video_idx)
            self.init_writer()
            with torch.no_grad():
               
                for idx, batch in enumerate(tqdm(self.dataloader)):
                    if batch is None:
                        continue
                    if self.tta is not None:
                        preds = self.tta.make_tta_predictions(self.model, batch)
                    else:
                        preds = self.model.inference_step(batch)

                    for idx, outputs in enumerate(preds):
                        ori_img = batch['ori_imgs'][idx]
                        img_w = batch['image_ws'][idx]
                        img_h = batch['image_hs'][idx]
                        img_ori_ws = batch['image_ori_ws'][idx]
                        img_ori_hs = batch['image_ori_hs'][idx]
                        
                        outputs = postprocessing(
                            outputs, 
                            current_img_size=[img_w, img_h],
                            ori_img_size=[img_ori_ws, img_ori_hs],
                            min_iou=self.min_iou,
                            min_conf=self.min_conf,
                            max_dets=self.max_dets,
                            keep_ratio=self.keep_ratio,
                            output_format='xywh',
                            mode=self.fusion_mode)

                        boxes = outputs['bboxes'] 
                        labels = outputs['classes']  
                        scores = outputs['scores']

                        if self.viz:
                        # Display all images in batch and write to video   
                            write_to_video(
                                ori_img, 
                                boxes=boxes,
                                labels=labels,
                                scores=scores,
                                imshow=False, 
                                outvid=self.outvid, 
                                obj_list=self.class_names)
                        



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

    pipeline = VideoDetect(args, config)
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
    config = get_config(args.weight, ignore_keys)
    if config is None:
        print("Config not found. Load configs from configs/configs.yaml")
        config = Config(os.path.join('configs','configs.yaml'))
    else:
        print("Load configs from weight")                 
    main(args, config)
    