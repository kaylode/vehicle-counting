# Pytorch Custom Template for Object Detection

## To-do list:
- [ ] Autoanchors
- [ ] Gradient checkpointing
- [ ] Distributed data parallel
- [ ] Sync BatchNorm
- [x] Multi-scale training (only works for YOLOv5)
- [x] Multi-GPU support (nn.DataParallel)
- [x] Cutmix, Mixup, strong augmentations
- [x] Test time augmentation
- [x] Gradient Accumulation
- [x] Mixed precision

## Dataset Structure:
```
this repo
│   detect.py
│   train.py
│
└───configs
│      configs.yaml
│
└───data  
│   └───<dataset's name>
│       └───images
│           │     00000.jpg
│           │     00001.jpg
│           │     ...
│       └───annotations
│           │     instances_train.json
│           │     instances_val.json
```

## JSON annotations file format:
Details can be found in: https://cocodataset.org/#format-data
```
{
  "images": 
  [
    {
      "id": int, 
      "width": int, 
      "height": int, 
      "file_name": str
    }, ...
  ],

"annotations" : 
  [
    {
      "id": int, 
      "image_id": int, 
      "category_id": int, 
      "area": float, 
      "bbox": [x_topleft,y_topleft,width,height], 
    }, ...
  ],

  "categories": 
  [
    {
      "id": int, 
      "name": str, 
      "supercategory": str,
    }, ...
  ]
}
```

## Configuration for custom dataset:
Open file configs/configs.yaml, and edit
```
settings:
  project_name: <dataset's name>    # also the folder name of the dataset that under ./data folder
  train_imgs: images/train
  val_imgs: images/val
  test_imgs: 
  train_anns: annotations/instances_train.json    # class index must start from 1
  val_anns: annotations/instances_val.json        # class index must start from 1
```

## Reference:
- Efficientdet from https://github.com/rwightman/efficientdet-pytorch
- FasterRCNN from torchvision
- Scaled YOLOv4 from https://github.com/WongKinYiu/ScaledYOLOv4
- YOLOv5 from https://github.com/ultralytics/yolov5
- Box fusion ensemble from https://github.com/ZFTurbo/Weighted-Boxes-Fusion
