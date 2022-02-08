import os
from tqdm import tqdm
from .detect import ImageDetect
from .track import VideoTracker, VideoCounting
from .datasets import VideoWriter, VideoLoader

class CountingPipeline:
    def __init__(self, args, config, cam_config):
        self.detector = ImageDetect(args, config)
        self.class_names = self.detector.class_names
        self.video_path = args.input_path
        self.saved_path = args.output_path
        self.cam_config = cam_config
        self.zone_path = cam_config.zone_path
        self.config = config

        if os.path.isdir(self.video_path):
            video_names = sorted(os.listdir(self.video_path))
            self.all_video_paths = [os.path.join(self.video_path, i) for i in video_names]
        else:
            self.all_video_paths = [self.video_path]

    def get_cam_name(self, path):
        filename = os.path.basename(path)
        cam_name = filename[:-4]
        return cam_name

    def run(self):
        for video_path in self.all_video_paths:
            cam_name = self.get_cam_name(video_path)
            videoloader = VideoLoader(self.config, video_path)
            self.tracker = VideoTracker(
                len(self.class_names),
                self.cam_config.cam[cam_name],
                videoloader.dataset.video_info,
                deepsort_chepoint=self.cam_config.checkpoint)

            videowriter = VideoWriter(
                videoloader.dataset.video_info,
                saved_path=self.saved_path,
                obj_list=self.class_names)
            
            videocounter = VideoCounting(
                class_names = self.class_names,
                zone_path = os.path.join(self.zone_path, cam_name+".json"))

            obj_dict = {
                'frames': [],
                'tracks': [],
                'labels': [],
                'boxes': []
            }

            for idx, batch in enumerate(tqdm(videoloader)):
                if batch is None:
                    continue
                preds = self.detector.run(batch)
                ori_imgs = batch['ori_imgs']

                for i in range(len(ori_imgs)):
                    boxes = preds['boxes'][i]
                    labels = preds['labels'][i]
                    scores = preds['scores'][i]
                    frame_id = batch['frames'][i]

                    ori_img = ori_imgs[i]

                    if len(boxes) == 0:
                        continue
                    track_result = self.tracker.run(ori_img, boxes, labels, scores)
                    

                    # box_xywh = change_box_order(track_result['boxes'],'xyxy2xywh');
                    # videowriter.write(
                    #     ori_img,
                    #     boxes = track_result['boxes'],
                    #     labels = box_xywh,
                    #     tracks = track_result['tracks'])
                    
                    for j in range(len(track_result['boxes'])):
                        obj_dict['frames'].append(frame_id)
                        obj_dict['tracks'].append(track_result['tracks'][j])
                        obj_dict['labels'].append(track_result['labels'][j])
                        obj_dict['boxes'].append(track_result['boxes'][j])

            

            result_dict = videocounter.run(
                    frames = obj_dict['frames'],
                    tracks = obj_dict['tracks'], 
                    labels = obj_dict['labels'],
                    boxes = obj_dict['boxes'],
                    output_path=os.path.join(self.saved_path, cam_name+'.csv'))

            videoloader.reinitialize_stream()
            videowriter.write_full_to_video(
                videoloader,
                num_classes = len(self.class_names),
                csv_path=os.path.join(self.saved_path, cam_name+'.csv'),
                paths=videocounter.directions,
                polygons=videocounter.polygons)