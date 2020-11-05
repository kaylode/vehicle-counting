from utils.getter import *

import json
import os

import argparse
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.backbone import EfficientDetBackbone
from utils.utils import postprocessing

WIDTH, HEIGHT = (1280, 720)

def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def main(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_transforms = get_augmentation(config, types = 'val')
    retransforms = Compose([
        Denormalize(box_transform=False),
        ToPILImage(),
        Resize(size = (WIDTH,HEIGHT))
    ])

    NUM_CLASSES = len(config.obj_list)
    net = EfficientDetBackbone(num_classes=NUM_CLASSES, compound_coef=args.compound_coef,
                                 ratios=eval(config.anchors_ratios), scales=eval(config.anchors_scales))
    model = Detector(
                    n_classes=NUM_CLASSES,
                    model = net,
                    criterion= FocalLoss(), 
                    optimizer= torch.optim.Adam,
                    optim_params = {'lr': 0.1},     
                    device = device)
    model.eval()

    if args.weight is not None:                
        load_checkpoint(net, args.weight)
    
    ann_path = os.path.join('datasets', config.project_name, config.val_anns)
    img_path = os.path.join('datasets', config.project_name, config.val_imgs)
    coco_gt = COCO(ann_path)
    image_ids = coco_gt.getImgIds()[:args.max_images]

    results = []
    img_id_list = []
    ims = []
    with torch.no_grad():
        for idx, image_id in enumerate(tqdm(image_ids)):
            img_id_list.append(image_id)
            image_info = coco_gt.loadImgs(image_id)[0]
            image_path = os.path.join(img_path,image_info['file_name'])
            img = val_transforms(Image.open(image_path).convert('RGB'))
            ims.append(img)

            if idx % args.batch_size == 0:
                batch = {'imgs': torch.stack([i['img'] for i in ims]).to(device)}
                ims = []
                preds = model.inference_step(batch, args.min_conf, args.min_iou)
                
                try:
                    preds = postprocessing(preds, batch['imgs'].cpu()[0], retransforms, out_format='xywh')
                except:
                    continue
                batch = None
            
                for pred, image_id in zip(preds, img_id_list):
                    cls_conf = pred['scores']
                    cls_ids = pred['classes']
                    bbox_xywh = pred['bboxes']
                    
                    for i in range(bbox_xywh.shape[0]):
                        score = float(cls_conf[i])
                        label = int(cls_ids[i])
                        box = bbox_xywh[i, :]
                        image_result = {
                            'image_id': image_id,
                            'category_id': label + 1,
                            'score': float(score),
                            'bbox': box.tolist(),
                        }

                        results.append(image_result)
                img_id_list = []
                

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'results/{config.project_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

    _eval(coco_gt, image_ids, filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training EfficientDet')
    parser.add_argument('--config' , type=str, help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('--max_images' , type=int, help='max number of images', default=10000)
    parser.add_argument('--weight' , type=str, help='project file that contains parameters')
    parser.add_argument('--min_conf', type=float, default= 0.3, help='minimum confidence for an object to be detect')
    parser.add_argument('--min_iou', type=float, default=0.3, help='minimum iou threshold for non max suppression')
    parser.add_argument('--batch_size', type=int, default=4, help='minimum iou threshold for non max suppression')

    args = parser.parse_args()
    config = Config(os.path.join('configs',args.config+'.yaml'))
    main(args, config)