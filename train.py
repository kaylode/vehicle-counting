from utils.getter import *
import argparse
import os
from datetime import datetime


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
seed_everything()

def train(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    devices_info = get_devices_info(config.gpu_devices)
    
    trainset, valset, trainloader, valloader = get_dataset_and_dataloader(config)
  
    net = get_model(args, config, device, num_classes=trainset.num_classes)

    if config.tta:
        config.tta = TTA(
            min_conf=config.min_conf_val, 
            min_iou=config.min_iou_val, 
            postprocess_mode=config.tta_ensemble_mode)
    else:
        config.tta = None

    metric = mAPScores(
        dataset=valset,
        min_conf = config.min_conf_val,
        min_iou = config.min_iou_val,
        tta=config.tta,
        mode=config.fusion_mode,
        max_dets=config.max_post_nms,
        keep_ratio=config.keep_ratio,
        max_images=config.max_images_val)

    optimizer, optimizer_params = get_lr_policy(config.lr_policy)

    model = Detector(
            model = net,
            metrics=metric,
            scaler=NativeScaler(),
            optimizer= optimizer,
            optim_params = optimizer_params,     
            device = device)

    if args.resume is not None:                
        load_checkpoint(model, args.resume)
        start_epoch, start_iter, best_value = get_epoch_iters(args.resume)
    else:
        print('Not resume. Initialize weights')
        start_epoch, start_iter, best_value = 0, 0, 0.0
        
    scheduler, step_per_epoch = get_lr_scheduler(
        model.optimizer, 
        lr_config=config.lr_scheduler,
        num_epochs=config.num_epochs)

    
    args.saved_path = os.path.join(
        args.saved_path, 
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    trainer = Trainer(config,
                     model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_iter=args.save_interval, path = args.saved_path),
                     best_value=best_value,
                     logger = Logger(log_dir=args.saved_path),
                     scheduler = scheduler,
                     evaluate_per_epoch = args.val_interval,
                     visualize_when_val = args.no_visualization,
                     step_per_epoch = step_per_epoch)
    print()
    print("##########   DATASET INFO   ##########")
    print("Trainset: ")
    print(trainset)
    print("Valset: ")
    print(valset)
    print()
    print(trainer)
    print()
    print(config)
    print(f'Training with {num_gpus} gpu(s): ')
    print(devices_info)
    print(f"Start training at [{start_epoch}|{start_iter}]")
    print(f"Current best MAP: {best_value}")
    
    trainer.fit(start_epoch = start_epoch, start_iter = start_iter, num_epochs=config.num_epochs, print_per_iter=args.print_per_iter)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training EfficientDet')
    parser.add_argument('--print_per_iter', type=int, default=300, help='Number of iteration to print')
    parser.add_argument('--val_interval', type=int, default=2, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=1000, help='Number of steps between saving')
    parser.add_argument('--resume', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize')
    parser.add_argument('--saved_path', type=str, default='./weights')
    parser.add_argument('--no_visualization', action='store_false', help='whether to visualize box to ./sample when validating (for debug), default=on')
    parser.add_argument('--freeze_backbone', action='store_true', help='whether to freeze the backbone')
    parser.add_argument('--freeze-bn', action='store_true', help='whether to freeze the backbone')
    
    args = parser.parse_args()
    config = Config(os.path.join('configs','configs.yaml'))

    train(args, config)
    


