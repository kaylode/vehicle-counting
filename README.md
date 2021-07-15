# Vehicle Tracking/Counting using YOLOv5 + DeepSORT
## Method
- Use [YOLOv5](https://github.com/ultralytics/yolov5) for vehicle detection task, only considers objects in Region of Interest (ROI)
- Use [DeepSORT](https://arxiv.org/abs/1703.07402) for car tracking, not need to retrain this model, only inference
- Use Cosine Similarity to assign object's tracks to most similar directions.
- Count each type of vehicle on each direction.

## Notebook
- For inference, use this notebook [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15pgDMnvXa-ZgGMeZkbbpg-gqa5Nttfi3?usp=sharing)
- To retrain detection model, follow instructions from [my template](https://github.com/kaylode/custom-template/tree/detection)

## Results
| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" alt="screen" src="demo/cam_09.gif"> | <img width="1604" alt="screen" src="demo/cam_10.gif"> | 
|<img width="1604" alt="screen" src="demo/cam_07.gif"> | <img width="1604" alt="screen" src="demo/cam_08.gif">|

-----------------------------------------------------------

## Dataset
- AIC-HCMC-2020: [link](https://drive.google.com/file/d/1iu4ifOTqnH_t80mL5IGasM6yKxhziPdL/view?usp=sharing)
<div align="center"><img width="1000" alt="screen" src="demo/dataset.png"></div>

## Pretrained weights
- Download finetuned YOLOv5 from on AIC-HCMC-2020 dataset:

Model | Image Size | Weights | MAP-S | MAP-M | MAP-L | ALL
--- | --- | --- | --- | --- | --- | ---
YOLOv5s | 640x640 | [link](https://drive.google.com/file/d/1urU0JKQqrI-BNtOCwqLSPr-aue1qWN0U/view?usp=sharing) | 0.068 | 0.312 | 0.390 | 0.32

## **Inference**

- File structure
```
this repo
│   detect.py
└───configs
│      configs.yaml           # Contains model's configurations
│      cam_configs.yaml       # Contains DEEPSORT's configuration for each video
```

- Install dependencies by ```pip install -r requirements.txt```
- To run full pipeline:
```
python detect.py --input_path=<input video or dir> --output_path=<output dir> --weight=<trained weight>
```
- **Extra Parameters**:
    - ***--min_conf***:     minimum confident for detection
    - ***--min_iou***:      minimum iou for detection

## References
- DeepSORT from https://github.com/ZQPei/deep_sort_pytorch
- YOLOv5 from https://github.com/ultralytics/yolov5
- Train YOLOv5 using https://github.com/kaylode/custom-template/tree/detection
