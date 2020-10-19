# Vehicle Counting using EfficientDet + DeepSORT (Pytorch)

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
- Use EfficientDet for vehicle detection task, finetune the model on the labeled dataset using ***train.py***
- Use DeepSORT for car tracking, not need to retrain this model, only inference

***Inference***
- Use trained detection model to detect vehicle in the video using ***detect.py***, all the bounding boxes, classes prediction will be saved to json files
- Use pretrained tracking model to track vehicle by their results from detection model above, use ***track.py*** 

# Finetuning EfficientDet:
```
python train.py 
```
- -c: Version number of EfficientDet, 0 - 7 
- --config: path to project config yaml file
- --resume: path to checkpoint to resume training
- --batch_size: batch size, recommend 4 - 8
- --head_only: if train only the head
- --num_epochs= number of epochs
- --saved_path= path to save weight


# Inference on AIC-HCMC testset:
***Run detection for detecting bounding boxes and classes confidence scores***
```
python aic-detect.py --path=datasets/aic-hcmc2020/video/cam_05.mp4 --batch_size=4 --coef=6 --min_conf=0.25 --min_iou=0.5 --weight=weights/efficientdet-d6_5_23204.pth --saved_path='/content/drive/My Drive/aic/boxes'
```
***Run tracking on detected bounding boxes and classes confidence scores***
```
python aic-tracker.py --ann_path='datasets/aic-hcmc2020/annotations' --box_path='/content/drive/My Drive/aic/boxes/cam_12' --img_path=datasets/aic-hcmc2020/frames/cam_12 --output=cam_12_det_10.mp4
```
# Results:

![Alt Text](results/cam09demo.gif)
![Alt Text](results/cam10_demo.gif)

# References:
- DeepSORT from https://github.com/nwojke/deep_sort
- EfficientDet from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
- AIC-HCMC Baseline: https://github.com/hcmcaic/ai-challenge-2020
