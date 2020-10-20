# Vehicle Counting/Tracking using EfficientDet + DeepSORT (Pytorch)

- Source code of ***#Team 059*** for AI Challenge of Ho Chi Minh City 2020
  - Task info: http://aichallenge.hochiminhcity.gov.vn/huong-dan-nhom-1
  - Leaderboard: http://aichcmc.ml/

# Demo
[Google Colab Tutorial](./demo/AIC_HCMC.ipynb)

# File Structure:
```
this repo
│   detect.py
│   track.py
│   train.py
│
└───datasets  
│   │
│   └───aic-hcmc2020
│       │
│       └───images
│       │     00000.jpg
│       │     00001.jpg
│       │     ....
│       └───annotations
│       │     instances_train.json
│       │     instances_val.json
│       │
│    └───annotations
│    |      cam_01.json
│    └───videos
│        │   
│        │  cam_01.mp4
│        │  cam_02.mp4
│        │  ...
│           
```

# Pretrained weights:
- Download pretrained EfficientDet from [original repo](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)


# Method:
***Training***
- Split the video into frames using ***preprocess.py***
- To manually label, use [labelme](https://github.com/wkentaro/labelme)
- Download all pretrained weights from above
- Use [EfficientDet](https://arxiv.org/abs/1911.09070) for vehicle detection task, finetune the model on the labeled dataset using ***train.py***
- Use [DeepSORT](https://arxiv.org/abs/1703.07402) for car tracking, not need to retrain this model, only inference

***Inference***
- Use trained detection model to detect vehicle in the video using ***detect.py***, all the bounding boxes, classes prediction will be saved to json files
- Use pretrained tracking model to track vehicle by their results from detection model above, use ***track.py*** 

# Finetuning EfficientDet:
```
python train.py -c=<version number of EfficientDet> --config=<path to project config yaml file>
```
- **Extra Parameters**:
    - ***--resume***:     path to checkpoint to resume training
    - ***--batch_size***: batch size, recommend 4 - 8
    - ***--head_only***:  if train only the head
    - ***--num_epochs***: number of epochs
    - ***--saved_path***: path to save weight
    - ***--val_interval***: validate per number of epochs
    - ***--save_interval***: save per number of iterations
    - ***--log_path***:     tensorboard logging path 

# Inference on AIC-HCMC testset:
***Run detection for detecting bounding boxes and classes confidence scores***
```
python detect.py -c=<version of EfficientDet> --path=<path to .mp4 video>
```
- **Extra Parameters**:
    - ***--min_conf***:     minimum confident for detection
    - ***--batch_size***:   batch size, recommend 4 - 8
    - ***--min_iou***:      minimum iou for detection
    - ***--weight***:       pretrained weight
    - ***--saved_path***:   path to save detection results

***Run tracking on detected bounding boxes and classes confidence scores***
```
python tracker.py --config=<cam config> --ann_path=<path to zones folder> --box_path=<path to detection results folder> --video_path=<path to .mp4 video>
```
- **Extra Parameters**:
    - ***--debug***:        for logging vehicle id
    - ***--output_path***:   path for output video and counting results
    
# Results:

![Alt Text](results/cam09demo.gif)
![Alt Text](results/cam10_demo.gif)

# References:
- DeepSORT from https://github.com/nwojke/deep_sort
- EfficientDet from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
- AIC-HCMC Baseline: https://github.com/hcmcaic/ai-challenge-2020
