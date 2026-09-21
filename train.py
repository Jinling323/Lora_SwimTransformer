from utils.regression_trainer_cosine_multibatch import RegTrainer
import argparse
import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--model-name', default='swin_large_trans', help='the name of the model')
    parser.add_argument('--stage', choices=['baseline', 'lora'], default='baseline',
                        help='baseline trains on clean; lora trains on hazy and validates on mix')
    parser.add_argument('--data-dir', '--clean-data-dir', dest='data_dir',
                        default='/media/mmslab5090/SSD2/crowd counting test/sha/clean',
                        help='baseline dataset root containing train/ and val/')
    parser.add_argument('--train-dir', '--mix-data-dir', dest='train_dir',
                        default='/media/mmslab5090/SSD2/crowd counting test/sha/hazy',
                        help='LoRA training root containing train/')
    parser.add_argument('--val-dir',
                        default='/media/mmslab5090/SSD2/crowd counting test/sha/mix',
                        help='LoRA validation root containing val/')
    parser.add_argument('--baseline-checkpoint', default='',
                        help='baseline best_model.pth or epoch checkpoint for LoRA initialization')
    parser.add_argument('--save-dir', default='model',
                        help='directory to save models.')
    parser.add_argument('--save-all', type=bool, default=False,
                        help='whether to save all best model')
    parser.add_argument('--lr', type=float, default=5*1e-6,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1200,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=600,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--swin-pretrained',
                        default='pre_models/swin_large_patch4_window12_384_22k.pth',
                        help='Swin-Large-Patch4-Window12 ImageNet-22K checkpoint')
    parser.add_argument('--lora-rank', type=int, default=8,
                        help='LoRA rank for Swin qkv/proj layers')
    parser.add_argument('--lora-alpha', type=float, default=16.0,
                        help='LoRA scaling alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.0,
                        help='reserved LoRA dropout (fused Swin attention requires 0)')
    parser.add_argument('--consistency-weight', type=float, default=1.0,
                        help='weight of the consistency loss')

    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=16,
                        help='downsample ratio')

    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')
    args = parser.parse_args()
    if args.model_name == 'swin_large_trans':
        if args.stage == 'baseline':
            if not args.data_dir:
                parser.error('--data-dir is required for baseline stage')
            if not args.swin_pretrained or not os.path.isfile(args.swin_pretrained):
                parser.error('baseline requires an existing --swin-pretrained checkpoint')
            args.train_dir = args.data_dir
            args.val_dir = args.data_dir
        else:
            if not args.train_dir or not args.val_dir:
                parser.error('lora requires --train-dir and --val-dir')
            if not args.baseline_checkpoint or not os.path.isfile(args.baseline_checkpoint):
                parser.error('lora requires an existing --baseline-checkpoint')
        for root, split in ((args.train_dir, 'train'), (args.val_dir, 'val')):
            path = os.path.join(root, split)
            if not os.path.isdir(path):
                parser.error('missing dataset directory: {}'.format(path))
        if args.crop_size <= 0 or args.crop_size % 32 != 0:
            parser.error('Swin crop size must be a positive multiple of 32')
        if args.resume and args.stage == 'lora' and args.baseline_checkpoint == args.resume:
            parser.error('--resume must be a LoRA-stage checkpoint, not the baseline checkpoint')
        args.save_dir = os.path.join(args.save_dir, args.stage)
    else:
        args.train_dir = args.data_dir
        args.val_dir = args.data_dir
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    set_seed(args.seed)
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
