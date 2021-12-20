# 🏍️ 🚙 Vehicle Tracking using YOLOv5 + DeepSORT 🚌


<details open> <summary><strong>Dev logs</strong></summary>
 <strong><i>[19/12/2021]</i></strong> Update to new YOLOv5 version 6. Can load checkpoints from original repo now 🤞 <br>
 <strong><i>[16/07/2021]</i></strong> BIG REFACTOR Code is cleaned and working fine now, promise 🤞 <br>
 <strong><i>[27/09/2021]</i></strong> All trained checkpoints on AIC-HCMC-2020 have been lost. Now use pretrained models on COCO for inference. 
</details>

## Method
- Use [YOLOv5](https://github.com/ultralytics/yolov5) for vehicle detection task, only considers objects in Region of Interest (ROI)
- Use [DeepSORT](https://arxiv.org/abs/1703.07402) for car tracking, not need to retrain this model, only inference
- Use Cosine Similarity to assign object's tracks to most similar directions.
- Count each type of vehicle on each direction.

## 📔 Notebook
- For inference, use this notebook [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15pgDMnvXa-ZgGMeZkbbpg-gqa5Nttfi3?usp=sharing)
- To retrain detection model, follow instructions from [original Yolov5](https://github.com/ultralytics/yolov5)

-----------------------------------------------------------

## Dataset
- AIC-HCMC-2020: [link](https://drive.google.com/drive/folders/15C6u58NDE8WQQSFflvhT_VEbLOtCG33A?usp=sharing)
- Direction and ROI annotation format:
```
cam_01.json # match video name
{
    "shapes": [
        {
            "label": "zone",
            "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4], ... ] #Points of a polygon
        },
        {
            "label": "direction01",
            "points": [[x1,y1], [x2,y2]] #Points of vector
        },
        {
            "label": "direction{id}",
            "points": [[x1,y1], [x2,y2]]
        },...
    ],
}
```

<div align="center"><img width="1000" alt="screen" src="demo/dataset.png"></div>

## 🥇 Pretrained weights
- Download finetuned models from on AIC-HCMC-2020 dataset:

Model | Image Size | Weights | Precision | Recall | MAP@0.5 | MAP@0.5-0.95
--- | --- | --- | --- | --- | --- | ---
YOLOv5s | 640x640 | [link](https://drive.google.com/file/d/1-Y6H3QdRevfBKYDQxgRiR2CRinRVPt9O/view?usp=sharing) | 0.87203 |	0.87356 |	0.91797 |	0.60795
YOLOv5m | 1024x1024 | [link](https://drive.google.com/file/d/10It3-bByVQUiLV9q4sdJDXQ3bNK9obKi/view?usp=sharing) | 0.89626	| 0.91098 |	0.94711 |	0.66816

## 🌟 **Inference**

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
python run.py --input_path=<input video or dir> --output_path=<output dir> --weight=<trained weight>
```
- **Extra Parameters**:
    - ***--min_conf***:     minimum confident for detection
    - ***--min_iou***:      minimum iou for detection

-----------------------------------------------------------

## Results

- After running, a .csv file contains results has following example format:

track_id |	frame_id |	box	| color |	label |	direction |	fpoint |	lpoint |	fframe |	lframe
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
2	| 3	| [607, 487, 664, 582]	| (144, 238, 144) |	0	| 1	| (635.5, 534.5)	| (977.0, 281.5)	| 3	| 109
2	| 4	| [625, 475, 681, 566]	| (144, 238, 144)	| 0	| 1	| (635.5, 534.5)	| (977.0, 281.5)	| 3	| 109
2	| 5	| [631, 471, 686, 561]	| (144, 238, 144)	| 0	| 1	| (635.5, 534.5)	| (977.0, 281.5)	| 3	| 109

- With:
  - `track_id`: the id of the object
  - `frame_id`: the current frame
  - `box`: the box wraps around the object in the corresponding frame
  - `color`: the color which is used to visualize the object
  - `direction`: the direction of the object
  - `fpoint`, `lpoint`: first/last coordinate where the object appears 
  - `fframe`, `lframe`: first/last frame where the object appears 
  
| Visualization result |
|:-------------------------:|
|<img width="1604" alt="screen" src="demo/cam_04.gif">|
|<img width="1604" alt="screen" src="demo/cam_07.gif">|


## References
- DeepSORT from https://github.com/ZQPei/deep_sort_pytorch
- YOLOv5 from https://github.com/ultralytics/yolov5
