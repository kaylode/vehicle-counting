from utils.getter import *
from models.detection.efficientdet.detector import EfficientDetector
from models.detection.efficientdet.model import FocalLoss
train_transforms = Compose([
    RandomHorizontalFlip(),
    Rotation(20),
    RandomCrop(),
    Resize((512,512)),
    ToTensor(),
    Normalize(box_transform = False),
])

val_transforms = Compose([
    Resize((512,512)),
    ToTensor(),
    Normalize(),
])

if __name__ == "__main__":
    dataset_path = "datasets/datasets/VOC/"
    img_path = dataset_path + "images"
    ann_path = {
        "train": dataset_path + "annotations/pascal_train2012.json",
        "val": dataset_path + "annotations/pascal_val2012.json"}
   
    trainset = ObjectDetectionDataset(img_dir=img_path, ann_path = ann_path['train'],transforms= train_transforms)
    valset = ObjectDetectionDataset(img_dir=img_path, ann_path = ann_path['val'], transforms=val_transforms)
    print(trainset)
    print(valset)

    trainset.mode='xyxy'
    valset.mode='xyxy'
    
    NUM_CLASSES = len(trainset.classes)
    device = torch.device("cuda")
    print("Using", device)

    # Dataloader
    BATCH_SIZE = 1
    my_collate = trainset.collate_fn
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=False)
    
    criterion = FocalLoss
    optimizer = torch.optim.Adam
    #metrics = [AccuracyMetric(decimals=3)]
    
    model = EfficientDetector(
                    n_classes = NUM_CLASSES,
                    optim_params = {'lr': 1e-3},
                    criterion= criterion, 
                    optimizer= optimizer,
                    freeze=True,
                    pretrained='weights/pretrained/efficientdet-d0-fixed.pth',
                    #metrics=  metrics,
                    device = device)
    
    #load_checkpoint(model, "weights/ssd-voc/SSD300-10.pth")
    #model.unfreeze()
    trainer = Trainer(model,
                     trainloader, 
                     valloader,
#                     clip_grad = 1.0,
                     checkpoint = Checkpoint(save_per_epoch=5, path = 'weights/effdet'),
                     logger = Logger(log_dir='loggers/runs/effdet'),
                     scheduler = StepLR(model.optimizer, step_size=30, gamma=0.1),
                     evaluate_per_epoch = 2)
    
    print(trainer)
    
    
    
    trainer.fit(num_epochs=50, print_per_iter=10)
    

    # Inference
    """imgs, results = trainer.inference_batch(valloader)
    idx = 0
    img = imgs[idx]
    boxes = results[idx]['rois']
    labels = results[idx]['class_ids']
    trainset.visualize(img,boxes,labels)"""